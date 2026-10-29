## WebRTC signaling + TURN ephemeral credentials, with optional native endpoint.
## Usage: nim c -d:release --threads:on --mm:orc -r src/main.nim
import mummy, mummy/routers
import std/[json, locks, tables, strutils, algorithm, sequtils, times, os, base64, openssl]
when defined(rtcNative):
  import rtc_native

const WebPage = staticRead("../web/index.html")
const PatchPage = staticRead("../web/patches.html")
const MaxRooms = 1024
const MaxGroups = 1024
# Upper admission limit; NOT a claim that 4096 simultaneous trainers converge.
# A large room uses sparse topology notifications instead of O(N) broadcasts.
const MaxGroupMembers = 4096
const MaxStoredRooms = 4096
const MaxListedRooms = 256
# Each peer consumes one lobby WS plus up to 8 transient pair signaling WS.
# The operator must configure OS file descriptors/RAM for this many sockets.
const MaxClients = 65536
const MaxWsMessageBytes = 140_000
const MaxRoomManifestBytes = 4096
const MaxMessagesPerTenSeconds = 80
const MaxBytesPerTenSeconds = 512_000

type RateState = object
  since: float
  packets: int
  bytes: int

type Client = object
  room: string
  native: bool
  group: bool
  peerId: string
  when defined(rtcNative):
    session: NativeSession

var mutex: Lock
initLock(mutex)
var clients = initTable[WebSocket, Client]()
var rooms = initTable[string, seq[WebSocket]]()
var groups = initTable[string, seq[WebSocket]]()
var groupManifests = initTable[string, JsonNode]()
# All training groups use bounded-v2. No legacy full-roster fallback.
var groupRevisions = initTable[string, int64]()
var groupDegree = initTable[string, int]()
var roomPublished = initTable[string, bool]()
var roomCreated = initTable[string, int64]()
var roomEverJoined = initTable[string, bool]()
var rates = initTable[WebSocket, RateState]()

let appToken = getEnv("APP_TOKEN")
let allowedOrigin = getEnv("ALLOWED_ORIGIN")
let turnSecret = getEnv("TURN_SECRET")
let turnHost = getEnv("TURN_HOST", "localhost")
let turnPort = getEnv("TURN_PORT", "3478")
let bindAddr = getEnv("BIND_ADDR", "127.0.0.1")
let roomDataDir = getEnv("ROOM_DATA_DIR", "fedpara-rooms")
let publishSecret = getEnv("ROOM_PUBLISH_TOKEN")
let listenPort = parseInt(getEnv("PORT", "8080"))

proc getString(n:JsonNode, key:string):string =
  if n != nil and n.kind == JObject and n.hasKey(key) and n[key].kind == JString:
    result = n[key].getStr()

proc errorWs(ws: WebSocket, explanation:string) {.gcsafe.} =
  ws.send($(%*{"type": "error", "message": explanation}))

proc equalToken(a, b:string): bool =
  # Fixed-time for equal-length strings; secret never sent in clear on a public URL.
  if a.len != b.len: return false
  var diff = 0
  for i in 0 ..< a.len:
    diff = diff or (ord(a[i]) xor ord(b[i]))
  diff == 0

proc turnKey(username:string):string =
  if turnSecret.len == 0: return ""
  var digest: array[20, byte]
  var digestLen: cuint
  let p = HMAC(EVP_sha1(), cast[pointer](turnSecret.cstring), turnSecret.len.cint,
               username.cstring, username.len.csize_t,
               cast[cstring](addr digest[0]), addr digestLen)
  if p == nil or digestLen != 20: return ""
  var bytes = newString(20)
  copyMem(addr bytes[0], addr digest[0], 20)
  result = base64.encode(bytes)

proc iceServers():JsonNode =
  result = newJArray()
  result.add(%*{"urls": "stun:" & turnHost & ":" & turnPort})
  if turnSecret.len > 0:
    let username = $(getTime().toUnix() + 3600) & ":peer"
    let password = turnKey(username)
    result.add(%*{"urls": ["turn:" & turnHost & ":" & turnPort & "?transport=udp",
                            "turn:" & turnHost & ":" & turnPort & "?transport=tcp"],
                   "username": username, "credential": password})

proc validRoom(s:string):bool =
  if s.len < 16 or s.len > 64: return false
  for c in s:
    if not (c in {'a'..'z', 'A'..'Z', '0'..'9', '_', '-'}): return false
  true

proc validRoomManifest(doc: JsonNode): bool =
  # Schema-only metadata. Never execute custom code or fetch datasets on Mummy.
  if doc == nil or doc.kind != JObject or ($doc).len > MaxRoomManifestBytes: return false
  if not doc.hasKey("version") or doc["version"].kind != JInt: return false
  let ver = doc["version"].getInt()
  if ver != 1 and ver != 2: return false
  if not doc.hasKey("model") or not doc.hasKey("code"): return false
  if doc["model"].kind != JObject or doc["code"].kind != JObject: return false
  let mdl = doc["model"]
  let code = doc["code"]
  for key in ["id", "revision", "architecture", "description"]:
    if mdl.getString(key).len > 240: return false
  if mdl.getString("id").len == 0 or mdl.getString("revision").len == 0: return false
  if ver == 1:
    if not doc.hasKey("dataset") or doc["dataset"].kind != JObject: return false
    let ds = doc["dataset"]
    for key in ["name", "description", "license", "privacy"]:
      if ds.getString(key).len > 240: return false
    if ds.getString("privacy") notin ["local-only", "public", "licensed"]: return false
  else:
    if not doc.hasKey("datasets") or doc["datasets"].kind != JObject: return false
    let dsc = doc["datasets"]
    if not dsc.hasKey("sources") or not dsc.hasKey("seed"): return false
    if dsc["seed"].kind != JInt or dsc["seed"].getInt() < 0 or dsc["seed"].getInt() > 2147483647: return false
    let sources = dsc["sources"]
    if sources.kind != JArray or sources.len < 1 or sources.len > 12: return false
    for source in sources.items:
      if source.kind != JObject: return false
      for key in ["repo", "revision", "split"]:
        if source.getString(key).len == 0 or source.getString(key).len > 200: return false
      if not source.hasKey("config") or source["config"].kind notin {JNull,JString}: return false
      if not source.hasKey("columns") or source["columns"].kind != JObject: return false
      let cols = source["columns"]
      if cols.len < 1 or cols.len > 2: return false
      for key,val in cols.pairs:
        if key notin ["text", "input", "target", "label"]: return false
        if val.kind != JString or val.getStr().len == 0 or val.getStr().len > 80: return false
      if not source.hasKey("weight") or source["weight"].kind notin {JFloat,JInt}: return false
      let wt = (if source["weight"].kind == JInt: source["weight"].getInt().float
                else: source["weight"].getFloat())
      if wt <= 0.0 or wt > 100.0: return false
  if code.hasKey("sha256") and code["sha256"].kind != JNull:
    let digest = code.getString("sha256")
    if digest.len != 64: return false
    for c in digest:
      if c notin {'0'..'9','a'..'f'}: return false
  let link = code.getString("source")
  if link.len > 512: return false
  if link.len > 0:
    if not link.startsWith("https://") or not code.hasKey("sha256") or code["sha256"].kind != JString: return false
  true

# Immutable metadata is persisted once, not at every presence event. Store the
# random room ID only in owner-readable disk files; never list private rooms.
proc persistRoom(id: string, manifest: JsonNode, isPublic: bool,
                 everJoined: bool = false, createdAt: int64 = 0): bool =
  if not validRoom(id) or manifest == nil or not validRoomManifest(manifest): return false
  try:
    createDir(roomDataDir)
    setFilePermissions(roomDataDir, {fpUserRead, fpUserWrite, fpUserExec})
    let name = roomDataDir / (id & ".json")
    let temp = roomDataDir / (id & ".tmp")
    let created = (if createdAt > 0: createdAt else: getTime().toUnix())
    let record = %*{"room":id, "manifest":manifest,
                    "public":isPublic, "created_at":created,
                    "ever_joined":everJoined}
    writeFile(temp, $record)
    setFilePermissions(temp, {fpUserRead, fpUserWrite})
    moveFile(temp, name)
    return true
  except CatchableError as e:
    echo "room directory write failed: ", e.msg
    return false

proc loadRoomDirectory() =
  if not dirExists(roomDataDir): return
  for path in walkFiles(roomDataDir / "*.json"):
    if groupManifests.len >= MaxStoredRooms: break
    try:
      if getFileSize(path) > MaxRoomManifestBytes + 1024: continue
      let record = parseFile(path)
      let id = record.getString("room")
      if not validRoom(id) or extractFilename(path) != id & ".json": continue
      if not record.hasKey("manifest") or not validRoomManifest(record["manifest"]): continue
      if record.hasKey("public") and record["public"].kind != JBool: continue
      groupManifests[id] = record["manifest"]
      roomPublished[id] = record.hasKey("public") and record["public"].getBool()
      roomEverJoined[id] = (if record.hasKey("ever_joined") and record["ever_joined"].kind == JBool:
                              record["ever_joined"].getBool() else: true)
      roomCreated[id] = (if record.hasKey("created_at") and record["created_at"].kind == JInt:
                           record["created_at"].getInt().int64 else: 0'i64)
    except CatchableError:
      discard  # Do not crash the signaling service on a corrupt metadata file.

proc acceptBudget(ws:WebSocket, n:int):bool {.gcsafe.} =
  {.cast(gcsafe).}:
    withLock mutex:
      if ws notin rates:return false
      var meter=rates[ws]
      let now=epochTime()
      if now-meter.since >= 10.0:
        meter=RateState(since:now)
      inc meter.packets
      meter.bytes+=n
      rates[ws]=meter
      result=meter.packets <= MaxMessagesPerTenSeconds and
             meter.bytes <= MaxBytesPerTenSeconds

# Constant-degree four-ring overlay. Room roster churn changes O(degree)
# edges, not O(N) long-distance chord edges. The FNV64 ordering MUST match
# Python patchlib/topology.py exactly. This is a topology hash, NOT a MAC.
proc ringHash(peer: string, ring: int): uint64 =
  var h = 0xcbf29ce484222325'u64
  let source = "kagami-v49-ring-" & $ring & ":" & peer
  for c in source:
    h = (h xor uint64(ord(c))) * 0x100000001b3'u64
  h = h xor (h shr 33)
  h = h * 0xff51afd7ed558ccd'u64
  h = h xor (h shr 33)
  h = h * 0xc4ceb9fe1a85ec53'u64
  result = h xor (h shr 33)

proc allNeighbors(ids: seq[string], degree: int = 8): Table[string,seq[string]] =
  result = initTable[string,seq[string]]()
  for peer in ids:
    result[peer] = @[]
  if ids.len < 2: return
  if ids.len <= 32:
    for pos, peer in ids:
      for i in 0 ..< degree div 2:
        let step = [1, 2, 4, 8][i]
        for sign in [-1, 1]:
          let index = ((pos + sign * step) mod ids.len + ids.len) mod ids.len
          let candidate = ids[index]
          if candidate != peer and candidate notin result[peer]:result[peer].add(candidate)
      if ids.len == 16 and degree == 8:
        let candidate = ids[(pos+(if pos mod 2 == 0: 3 else: -3)+16) mod 16]
        if candidate != peer and candidate notin result[peer]:result[peer].add(candidate)
      result[peer].sort()
    return
  for ring in 0 ..< degree div 2:
    var order = ids
    if ring > 0:
      order = ids.sorted(proc(a,b:string):int =
        let ha=ringHash(a,ring)
        let hb=ringHash(b,ring)
        if ha < hb: -1
        elif ha > hb: 1
        else: cmp(a,b))
    for pos, peer in order:
      let left=order[(pos-1+order.len) mod order.len]
      let right=order[(pos+1) mod order.len]
      if left != peer and left notin result[peer]:result[peer].add(left)
      if right != peer and right notin result[peer]:result[peer].add(right)
  for peer in ids:
    result[peer].sort()

proc sortedNeighbors(ids: seq[string], peer: string, degree: int = 8): seq[string] =
  if peer notin ids: return @[]
  allNeighbors(ids,degree).getOrDefault(peer)

proc sortedPeers(sockets: seq[WebSocket]): seq[string] =
  for sock in sockets:
    if sock in clients:result.add(clients[sock].peerId)
  result.sort()

proc changedNeighborNotices(beforeIds, afterIds: seq[string], changed: string,
                            wsByPeer: Table[string,WebSocket],
                            version: int64, degree: int = 8): seq[tuple[target:WebSocket, payload:string]] =
  # Build O(N log N) deterministic ring overlays once per membership change.
  # A join/leave changes the two neighbors of a node in each ring; in steady
  # state this sends only a bounded number of per-peer notifications.
  let old=allNeighbors(beforeIds,degree)
  let updated=allNeighbors(afterIds,degree)
  for peer in afterIds:
    if peer notin wsByPeer: continue
    let nextPeers=updated.getOrDefault(peer)
    if peer notin old or old[peer]!=nextPeers:
      let notice = $(%*{"type":"group-neighbors", "topology":"bounded-v2",
                        "peers":nextPeers, "online":afterIds.len,
                        "revision":version})
      result.add((target:wsByPeer[peer],payload:notice))

proc removeClient(ws:WebSocket) {.gcsafe.} =
  {.cast(gcsafe).}:
    var peer: WebSocket
    var notify = false
    var departed = ""
    var topologyNotices:seq[tuple[target:WebSocket,payload:string]] = @[]
    when defined(rtcNative):
      var session:NativeSession = nil
    withLock mutex:
      if ws in rates: rates.del(ws)
      if ws in clients:
        let client = clients[ws]
        when defined(rtcNative):
          if client.native: session = client.session
        if client.group and client.room in groups:
          departed = client.peerId
          let oldIds=sortedPeers(groups[client.room])
          var kept: seq[WebSocket] = @[]
          var byPeer=initTable[string,WebSocket]()
          for candidate in groups[client.room]:
            if candidate != ws:
              kept.add(candidate)
              if candidate in clients:
                byPeer[clients[candidate].peerId]=candidate
          if kept.len == 0:
            groups.del(client.room)
            groupDegree.del(client.room)
            groupRevisions.del(client.room)
          else:
            groups[client.room] = kept
            let ver=groupRevisions.getOrDefault(client.room,0'i64)+1
            groupRevisions[client.room]=ver
            topologyNotices=changedNeighborNotices(
                oldIds,sortedPeers(kept),departed,byPeer,ver,
                groupDegree.getOrDefault(client.room,8))
        elif not client.native and client.room in rooms:
          var kept: seq[WebSocket] = @[]
          for candidate in rooms[client.room]:
            if candidate != ws:
              kept.add(candidate)
              peer = candidate
              notify = true
          if kept.len == 0: rooms.del(client.room)
          else: rooms[client.room] = kept
        clients.del(ws)
    when defined(rtcNative):
      if session != nil: destroyNativeSession(session)
    if notify: peer.send("""{"type":"peer-left"}""")
    for notice in topologyNotices:
      try: notice.target.send(notice.payload)
      except CatchableError: discard

proc groupRegister(ws:WebSocket, data:JsonNode) {.gcsafe.} =
  # Register metadata without joining or taking a peer slot. It does NOT mark
  # the room as trained; the first real group-join can still found its genesis.
  {.cast(gcsafe).}:
    if appToken.len > 0 and not equalToken(appToken, getString(data,"token")):
      errorWs(ws,"Invalid access token")
      return
    let room = getString(data,"room")
    if room.len < 24 or not validRoom(room):
      errorWs(ws,"Invalid room ID")
      return
    if not data.hasKey("manifest") or not validRoomManifest(data["manifest"]):
      errorWs(ws,"Invalid room manifest")
      return
    let publish = data.hasKey("publish") and data["publish"].kind == JBool and
                  data["publish"].getBool()
    if data.hasKey("publish") and data["publish"].kind != JBool:
      errorWs(ws,"Invalid publish setting")
      return
    if publish and (publishSecret.len > 0 and
                    not equalToken(publishSecret,getString(data,"publish_token"))):
      errorWs(ws,"Publishing requires ROOM_PUBLISH_TOKEN")
      return
    if publish and publishSecret.len == 0 and bindAddr notin ["127.0.0.1", "::1"]:
      errorWs(ws,"Publishing disabled without ROOM_PUBLISH_TOKEN")
      return
    var accepted=false
    withLock mutex:
      if ws notin clients:
        if room in groupManifests:
          accepted=(groupManifests[room]==data["manifest"] and
                    roomPublished.getOrDefault(room,false)==publish)
        elif groupManifests.len < MaxStoredRooms:
          var publicCount=0
          if publish:
            for entry in roomPublished.values:
              if entry:inc publicCount
          if not publish or publicCount < MaxListedRooms:
            if persistRoom(room,data["manifest"],publish,false):
              groupManifests[room]=data["manifest"]
              roomPublished[room]=publish
              roomCreated[room]=getTime().toUnix()
              roomEverJoined[room]=false
              accepted=true
    if accepted:
      ws.send($(%*{"type":"room-registered","room":room}))
    else:
      errorWs(ws,"Room already registered with different immutable metadata, or capacity reached")

proc groupDescribe(ws:WebSocket, data:JsonNode) {.gcsafe.} =
  # Read-only metadata lookup. Unlike group-join, this must NOT create
  # presence, modify roomEverJoined, or send group-peer-joined notifications.
  # Otherwise merely pressing the GUI's "load room" button consumes the
  # founder's one-time genesis entitlement and strands both GUI windows.
  {.cast(gcsafe).}:
    if appToken.len > 0 and not equalToken(appToken, getString(data, "token")):
      errorWs(ws, "Invalid server token")
      return
    let room = getString(data, "room")
    if room.len < 24 or not validRoom(room):
      errorWs(ws, "Invalid room ID")
      return
    var manifest: JsonNode = nil
    withLock mutex:
      if room in groupManifests:
        manifest = groupManifests[room]
    if manifest == nil:
      errorWs(ws, "Room metadata not found")
    else:
      ws.send($(%*{"type":"room-description","room":room,"manifest":manifest}))

proc groupJoin(ws:WebSocket, data:JsonNode) {.gcsafe.} =
  # Room ID is a bearer secret. Signaling server keeps only bounded presence.
  {.cast(gcsafe).}:
    if appToken.len > 0 and not equalToken(appToken,getString(data,"token")):
      errorWs(ws,"Invalid server token")
      ws.close()
      return
    let room=getString(data,"room")
    let identity=getString(data,"peer")
    # Required to prevent an old full-roster client from silently accepting
    # an eight-peer partial roster and corrupting its topology decisions.
    if not data.hasKey("topology_v2") or
        data["topology_v2"].kind != JBool or not data["topology_v2"].getBool():
      errorWs(ws,"Unsupported client: bounded-v2 topology is required; upgrade Fedpara")
      return
    let requestedDegree=(if data.hasKey("topology_degree") and
                            data["topology_degree"].kind == JInt:
                            data["topology_degree"].getInt() else: 8)
    if requestedDegree notin [2,4,6,8] or
        (data.hasKey("topology_degree") and
         data["topology_degree"].kind != JInt):
      errorWs(ws,"Invalid topology degree")
      return
    let manifest=(if data.hasKey("manifest"):data["manifest"] else:nil)
    let publish = data.hasKey("publish") and data["publish"].kind == JBool and
                  data["publish"].getBool()
    if data.hasKey("publish") and data["publish"].kind != JBool:
      errorWs(ws,"Invalid publish flag")
      return
    if publish and (publishSecret.len > 0 and
                    not equalToken(publishSecret, getString(data, "publish_token"))):
      errorWs(ws,"Room listing publication requires operator authorization")
      return
    if publish and publishSecret.len == 0 and bindAddr notin ["127.0.0.1", "::1"]:
      errorWs(ws,"Public listing disabled without ROOM_PUBLISH_TOKEN")
      return
    if manifest != nil and not validRoomManifest(manifest):
      errorWs(ws,"Invalid bounded room model/dataset/code descriptor")
      return
    if room.len < 24 or not validRoom(room):
      errorWs(ws,"Group room must be >=24 ASCII characters")
      return
    if identity.len != 33 or identity[0]!='p':
      errorWs(ws,"Invalid peer fingerprint")
      return
    for c in identity[1..^1]:
      if c notin {'0'..'9','a'..'f'}:
        errorWs(ws,"Invalid peer fingerprint")
        return
    var members:seq[string] = @[]
    var accepted=false
    var previouslyRegistered=false
    var incompatible=false
    var capacityReached=false
    var version=0'i64
    var occupancy=1
    var topologyNotices:seq[tuple[target:WebSocket,payload:string]] = @[]
    withLock mutex:
      previouslyRegistered = roomEverJoined.getOrDefault(room,false)
      if ws notin clients and clients.len < MaxClients and
          (room in groups or groups.len < MaxGroups):
        let existing=groups.getOrDefault(room)
        let degreeAllowed=(existing.len == 0 or
                           groupDegree.getOrDefault(room,8)==requestedDegree)
        if not degreeAllowed:incompatible=true
        let compatible=degreeAllowed and (room notin groupManifests or manifest == nil or
                        groupManifests[room] == manifest) and
                        (not publish or room notin groupManifests or
                         roomPublished.getOrDefault(room,false))
        if existing.len >= MaxGroupMembers: capacityReached=true
        if existing.len < MaxGroupMembers and compatible:
          var duplicate=false
          for prior in existing:
            if prior in clients and clients[prior].peerId==identity:
              duplicate=true
              break
          # Unknown rooms require metadata on first join. This prevents
          # silently creating a room which cannot later be listed/rejoined.
          if room notin groupManifests and manifest == nil: duplicate=true
          if room notin groupManifests and groupManifests.len >= MaxStoredRooms: duplicate=true
          if room notin groupManifests and publish:
            var publicCount=0
            for published in roomPublished.values:
              if published: inc publicCount
            if publicCount >= MaxListedRooms: duplicate=true
          if not duplicate and room notin groupManifests:
            if not persistRoom(room,manifest,publish,true): duplicate=true
            else:
              groupManifests[room]=manifest
              roomPublished[room]=publish
              roomCreated[room]=getTime().toUnix()
              roomEverJoined[room]=true
          if not duplicate and room in groupManifests and
              not roomEverJoined.getOrDefault(room,false):
            # A GUI-created room was saved without training. Its first REAL
            # peer may still found genesis; future joins must not reset it.
            if not persistRoom(room,groupManifests[room],
                               roomPublished.getOrDefault(room,false),true,
                               roomCreated.getOrDefault(room,0'i64)):
              duplicate=true
            else:
              roomEverJoined[room]=true
          if not duplicate:
            var oldIds:seq[string] = @[]
            var byPeer=initTable[string,WebSocket]()
            for prior in existing:
              if prior in clients:
                oldIds.add(clients[prior].peerId)
                byPeer[clients[prior].peerId]=prior
            groups.mgetOrPut(room,@[]).add(ws)
            clients[ws]=Client(room:room,group:true,peerId:identity)
            groupDegree[room]=requestedDegree
            oldIds.sort()
            var newIds=oldIds & @[identity]
            newIds.sort()
            occupancy=newIds.len
            version=groupRevisions.getOrDefault(room,0'i64)+1
            groupRevisions[room]=version
            members=sortedNeighbors(newIds,identity,requestedDegree)
            topologyNotices=changedNeighborNotices(oldIds,newIds,identity,
                                                   byPeer,version,requestedDegree)
            accepted=true
    if not accepted:
      errorWs(ws,(if incompatible: "Room topology version incompatible"
                  elif capacityReached: "Room participant capacity reached"
                  else: "Group full, identity already online, or server busy"))
      return
    var roster=newJArray()
    for id in members: roster.add(%id)
    var reply = %*{"type":"group-members","peers":roster,
                    "existing_room":previouslyRegistered}
    reply["topology"] = %"bounded-v2"
    reply["degree"] = %requestedDegree
    reply["online"] = %occupancy
    reply["revision"] = %version
    withLock mutex:
      if room in groupManifests:reply["manifest"]=groupManifests[room]
    ws.send($reply)
    for notice in topologyNotices:
      try: notice.target.send(notice.payload)
      except CatchableError: discard

proc onJoin(ws:WebSocket, data:JsonNode) {.gcsafe.} =
  {.cast(gcsafe).}:
    if appToken.len > 0 and not equalToken(appToken, getString(data, "token")):
      errorWs(ws, "Invalid access token")
      ws.close()
      return

    let mode = getString(data, "mode")
    if mode == "echo":
      when defined(rtcNative):
        var exists = false
        withLock mutex:
          exists = ws in clients
        if exists:
          errorWs(ws, "Already joined")
          return
        let session = createNativeSession(ws)
        if session == nil:
          errorWs(ws, "Failed to create native PeerConnection")
          return
        var withinLimit = false
        withLock mutex:
          if clients.len < MaxClients:
            clients[ws] = Client(native:true, session:session)
            withinLimit = true
        if not withinLimit:
          destroyNativeSession(session)
          errorWs(ws, "Server busy")
          return
        ws.send($(%*{"type":"joined", "role":"native-client", "iceServers":iceServers()}))
      else:
        errorWs(ws, "Native endpoint unavailable: compile with -d:rtcNative")
      return
    if mode != "p2p":
      errorWs(ws, "Mode must be 'p2p' or 'echo'")
      return

    let room = getString(data, "room")
    if not validRoom(room):
      errorWs(ws, "Room must be 16-64 characters [a-zA-Z0-9_-]")
      return
    var count = 0
    var joined = false
    var other: WebSocket
    withLock mutex:
      if ws notin clients and clients.len < MaxClients and (room in rooms or rooms.len < MaxRooms):
        count = rooms.getOrDefault(room).len
        if count < 2:
          rooms.mgetOrPut(room, @[]).add(ws)
          clients[ws] = Client(room:room, native:false)
          if count == 1: other = rooms[room][0]
          joined = true
    if not joined:
      errorWs(ws, "Already joined / room full / server busy")
      return
    ws.send($(%*{"type":"joined", "role": (if count == 0: "initiator" else: "answerer"),
                  "room":room, "iceServers":iceServers()}))
    if count == 1:
      other.send("""{"type":"peer-joined"}""")

proc forwardMessage(ws:WebSocket, data:JsonNode, kind:string) {.gcsafe.} =
  {.cast(gcsafe).}:
    var peer:WebSocket
    var found = false
    var native = false
    when defined(rtcNative):
      var session:NativeSession = nil
    withLock mutex:
      if ws in clients:
        let c = clients[ws]
        native = c.native
        when defined(rtcNative):
          if native: session = c.session
        if not c.native and not c.group and c.room in rooms and rooms[c.room].len == 2:
          let rs = rooms[c.room]
          peer = (if rs[0] == ws: rs[1] else: rs[0])
          found = true

    if native:
      when defined(rtcNative):
        if kind == "offer":
          let sdp = getString(data, "sdp")
          if sdp.len == 0 or sdp.len > 128_000 or not acceptOffer(session, sdp):
            errorWs(ws, "Native SDP offer rejected")
        elif kind == "candidate":
          let candObj = if data.hasKey("candidate"): data["candidate"] else: nil
          if candObj == nil or candObj.kind == JNull: return
          let candidate = getString(candObj, "candidate")
          let mid = getString(candObj, "sdpMid")
          if candidate.len > 2048 or mid.len > 128 or not acceptCandidate(session, candidate, mid):
            errorWs(ws, "Native ICE candidate rejected")
        else:
          errorWs(ws, "Native mode accepts offer and candidate only")
      return

    if not found:
      errorWs(ws, "No peer in room")
      return
    if kind == "offer" or kind == "answer":
      let sdp = getString(data, "sdp")
      if sdp.len == 0 or sdp.len > 128_000: return
      peer.send($(%*{"type":kind, "sdp":sdp}))
    elif kind == "candidate":
      if not data.hasKey("candidate"): return
      let c = data["candidate"]
      if c.kind == JNull:
        peer.send("""{"type":"candidate","candidate":null}""")
      elif c.kind == JObject:
        if getString(c, "candidate").len > 2048: return
        if getString(c, "sdpMid").len > 128: return
        peer.send($(%*{"type":"candidate", "candidate":c}))

proc websocketHandler(ws:WebSocket, event:WebSocketEvent, message:Message) {.gcsafe.} =
  case event
  of OpenEvent:
    {.cast(gcsafe).}:
      var accepted=false
      withLock mutex:
        if rates.len < MaxClients:
          rates[ws]=RateState(since:epochTime())
          accepted=true
      if not accepted:ws.close()
  of ErrorEvent: discard
  of CloseEvent: removeClient(ws)
  of MessageEvent:
    if not acceptBudget(ws,message.data.len):
      ws.close()
      return
    if message.kind == Ping:
      ws.send("", Pong)
      return
    if message.kind != TextMessage or message.data.len > MaxWsMessageBytes:
      errorWs(ws, "Invalid message")
      return
    var doc:JsonNode
    try:
      doc = parseJson(message.data)
    except JsonParsingError:
      errorWs(ws, "Invalid JSON")
      return
    if doc == nil or doc.kind != JObject:
      errorWs(ws, "Expected JSON object")
      return
    let kind = getString(doc, "type")
    case kind
    of "join": onJoin(ws, doc)
    of "group-join": groupJoin(ws, doc)
    of "group-describe": groupDescribe(ws, doc)
    of "room-register": groupRegister(ws, doc)
    of "offer", "answer", "candidate": forwardMessage(ws, doc, kind)
    of "leave":
      removeClient(ws)
      ws.close()
    of "ping":
      {.cast(gcsafe).}:
        var compact=false
        var count=0
        var version=0'i64
        withLock mutex:
          if ws in clients and clients[ws].group:
            let room=clients[ws].room
            compact=true
            count=groups.getOrDefault(room).len
            version=groupRevisions.getOrDefault(room,0'i64)
        if compact:
          ws.send($(%*{"type":"group-count","online":count,
                        "revision":version}))
        else:
          ws.send("""{"type":"pong"}""")
    else: errorWs(ws, "Unknown message type")

proc htmlHandler(request:Request) {.gcsafe.} =
  var headers:HttpHeaders
  headers["Content-Type"] = "text/html; charset=utf-8"
  headers["Cache-Control"] = "no-store"
  headers["X-Content-Type-Options"] = "nosniff"
  request.respond(200, headers, WebPage)

proc patchesHandler(request:Request) {.gcsafe.} =
  var headers:HttpHeaders
  headers["Content-Type"] = "text/html; charset=utf-8"
  headers["Cache-Control"] = "no-store"
  headers["X-Content-Type-Options"] = "nosniff"
  request.respond(200, headers, PatchPage)

proc healthHandler(request:Request) {.gcsafe.} =
  var headers:HttpHeaders
  headers["Content-Type"] = "application/json"
  request.respond(200, headers, """{"ok":true}""")

proc directoryHandler(request:Request) {.gcsafe.} =
  # Only explicitly published rooms are enumerable. Room IDs are bearer
  # secrets: opting into directory listing grants discoverability and entry.
  {.cast(gcsafe).}:
    var resultRows = newJArray()
    withLock mutex:
      for id, doc in groupManifests.pairs:
        if resultRows.len >= MaxListedRooms: break
        if not roomPublished.getOrDefault(id, false): continue
        var datasets = newJArray()
        if doc.hasKey("datasets") and doc["datasets"].kind == JObject and
            doc["datasets"].hasKey("sources"):
          for item in doc["datasets"]["sources"].items:
            datasets.add(%item.getString("repo"))
        elif doc.hasKey("dataset"):
          datasets.add(%doc["dataset"].getString("name"))
        let live = groups.getOrDefault(id).len
        resultRows.add(%*{"room_id":id,
                            "model":doc["model"].getString("id"),
                            "revision":doc["model"].getString("revision"),
                            "datasets":datasets,"online":live,
                            "capacity":MaxGroupMembers,
                            "created_at":roomCreated.getOrDefault(id,0'i64)})
    var headers:HttpHeaders
    headers["Content-Type"]="application/json; charset=utf-8"
    headers["Cache-Control"]="no-store"
    headers["X-Content-Type-Options"]="nosniff"
    request.respond(200,headers,$(%*{"rooms":resultRows,"limit":MaxListedRooms}))

proc upgradeHandler(request:Request) {.gcsafe.} =
  # ALLOWED_ORIGIN is read-only after initialization. Safe to share across workers.
  {.cast(gcsafe).}:
    if allowedOrigin.len > 0 and request.headers["Origin"] != allowedOrigin:
      request.respond(403)
      return
    discard request.upgradeToWebSocket()

if bindAddr != "127.0.0.1" and bindAddr != "::1" and appToken.len == 0:
  echo "Warning: public room-only signaling is enabled. Use a WSS reverse proxy, unguessable room IDs, and IP-level connection/rate limits."
if turnSecret.len > 0 and turnSecret.len < 32:
  quit("TURN_SECRET must have >= 32 characters", QuitFailure)

loadRoomDirectory()
var router:Router
router.get("/", htmlHandler)
router.get("/healthz", healthHandler)
router.get("/api/rooms", directoryHandler)
router.get("/patches", patchesHandler)
router.get("/ws", upgradeHandler)
# Version-compatible Mummy initialization; see scripts/check_nim.sh.
when compiles(newServer(router, websocketHandler,
                        workerThreads=4, maxMessageLen=MaxWsMessageBytes)):
  let server = newServer(router, websocketHandler,
                         workerThreads=4, maxMessageLen=MaxWsMessageBytes)
  echo "Mummy WebRTC server on ", bindAddr, ":", listenPort
  when defined(rtcNative):
    echo "Native libdatachannel endpoint: enabled"
  else:
    echo "Native endpoint: disabled"
  when compiles(server.serve(Port(listenPort), bindAddr)):
    server.serve(Port(listenPort), bindAddr)
  else:
    if bindAddr != "127.0.0.1":
      quit("Mummy does not support BIND_ADDR in this version", QuitFailure)
    server.serve(Port(listenPort))
else:
  let server = newServer(router, websocketHandler, workerThreads=4)
  echo "Warning: install updated Mummy or limit WebSocket frame size at reverse proxy."
  echo "Mummy WebRTC server on ", bindAddr, ":", listenPort
  when compiles(server.serve(Port(listenPort), bindAddr)):
    server.serve(Port(listenPort), bindAddr)
  else:
    if bindAddr != "127.0.0.1":
      quit("Mummy does not support BIND_ADDR in this version", QuitFailure)
    server.serve(Port(listenPort))
