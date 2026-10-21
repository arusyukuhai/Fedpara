## Optional libdatachannel C-API bridge. Build with -d:rtcNative.
## The browser sends an SDP offer with a DataChannel. We answer, accept
## trickled ICE candidates, and echo text/binary DataChannel frames.
## libdatachannel provides ICE, DTLS, SCTP, and media SRTP when tracks exist.
import mummy
import std/[json, strutils, atomics]

when defined(macosx):
  const rtcLib = "libdatachannel.dylib"
elif defined(windows):
  const rtcLib = "datachannel.dll"
else:
  const rtcLib = "libdatachannel.so(|.0)"

type
  RtcConfig {.bycopy.} = object
    iceServers: ptr cstring
    iceServersCount: cint
    proxyServer, bindAddress: cstring
    certificateType, iceTransportPolicy: cint
    enableIceTcp, enableIceUdpMux, disableAutoNegotiation, forceMediaTransport: bool
    portRangeBegin, portRangeEnd: uint16
    mtu, maxMessageSize: cint
  DescriptionCb = proc(pc:cint, sdp, kind:cstring, user:pointer) {.cdecl, gcsafe, raises: [].}
  CandidateCb = proc(pc:cint, cand, mid:cstring, user:pointer) {.cdecl, gcsafe, raises: [].}
  DataChannelCb = proc(pc, dc:cint, user:pointer) {.cdecl, gcsafe, raises: [].}
  TrackCb = proc(pc, tr:cint, user:pointer) {.cdecl, gcsafe, raises: [].}
  MessageCb = proc(dc:cint, data:cstring, size:cint, user:pointer) {.cdecl, gcsafe, raises: [].}
  StateCb = proc(pc:cint, state:cint, user:pointer) {.cdecl, gcsafe, raises: [].}
  NativeSession* = ptr NativeSessionObj
  NativeSessionObj = object
    socket: WebSocket
    pc: cint
    remoteSet: bool
    rtpPackets: Atomic[uint32]
    pending: seq[tuple[candidate, mid:string]]

proc rtcCreatePeerConnection(config: ptr RtcConfig):cint {.cdecl, importc, dynlib: rtcLib.}
proc rtcDeletePeerConnection(pc:cint):cint {.cdecl, importc, dynlib: rtcLib.}
proc rtcSetUserPointer(pc:cint, data:pointer) {.cdecl, importc, dynlib: rtcLib.}
proc rtcSetLocalDescriptionCallback(pc:cint, cb:DescriptionCb):cint {.cdecl, importc, dynlib: rtcLib.}
proc rtcSetLocalCandidateCallback(pc:cint, cb:CandidateCb):cint {.cdecl, importc, dynlib: rtcLib.}
proc rtcSetDataChannelCallback(pc:cint, cb:DataChannelCb):cint {.cdecl, importc, dynlib: rtcLib.}
proc rtcSetTrackCallback(pc:cint, cb:TrackCb):cint {.cdecl, importc, dynlib: rtcLib.}
proc rtcSetStateChangeCallback(pc:cint, cb:StateCb):cint {.cdecl, importc, dynlib: rtcLib.}
proc rtcSetMessageCallback(dc:cint, cb:MessageCb):cint {.cdecl, importc, dynlib: rtcLib.}
proc rtcSetRemoteDescription(pc:cint, sdp, kind:cstring):cint {.cdecl, importc, dynlib: rtcLib.}
proc rtcSetLocalDescription(pc:cint, kind:cstring):cint {.cdecl, importc, dynlib: rtcLib.}
proc rtcAddRemoteCandidate(pc:cint, cand, mid:cstring):cint {.cdecl, importc, dynlib: rtcLib.}
proc rtcSendMessage(dc:cint, data:cstring, size:cint):cint {.cdecl, importc, dynlib: rtcLib.}

# C library callbacks can run on foreign threads. Register those threads
# with Nim before allocating strings/JSON. Exception must not cross the C ABI.
proc ensureForeignThread() {.inline, gcsafe, raises: [].} =
  when declared(setupForeignThreadGc):
    setupForeignThreadGc()

proc sendLocalDescription(pc:cint, sdp, kind:cstring, user:pointer) {.cdecl, gcsafe, raises: [].} =
  ensureForeignThread()
  if user == nil or sdp == nil or kind == nil: return
  try:
    let session = cast[NativeSession](user)
    session.socket.send($(%*{"type": $kind, "sdp": $sdp}))
  except CatchableError:
    discard

proc sendLocalCandidate(pc:cint, cand, mid:cstring, user:pointer) {.cdecl, gcsafe, raises: [].} =
  ensureForeignThread()
  if user == nil: return
  try:
    let session = cast[NativeSession](user)
    if cand != nil and ($cand).len > 0:
      let m = if mid == nil: "" else: $mid
      session.socket.send($(%*{"type": "candidate", "candidate": {"candidate": $cand, "sdpMid": m}}))
    else:
      session.socket.send("""{"type":"candidate","candidate":null}""")
  except CatchableError:
    discard

proc echoMessage(dc:cint, data:cstring, size:cint, user:pointer) {.cdecl, gcsafe, raises: [].} =
  # size<0 means a NUL-terminated UTF-8 text message, otherwise binary.
  if data != nil:
    discard rtcSendMessage(dc, data, size)

proc onDataChannel(pc, dc:cint, user:pointer) {.cdecl, gcsafe, raises: [].} =
  discard rtcSetMessageCallback(dc, echoMessage)

proc onMediaRtp(tr:cint, data:cstring, size:cint, user:pointer) {.cdecl, gcsafe, raises: [].} =
  if user == nil or size <= 0: return
  let session = cast[NativeSession](user)
  let count = session.rtpPackets.fetchAdd(1'u32, moRelaxed) + 1'u32
  if count mod 250'u32 == 0'u32:
    ensureForeignThread()
    try:
      session.socket.send($(%*{"type":"rtp-stats", "receivedPackets":count}))
    except CatchableError:
      discard

proc onTrack(pc, tr:cint, user:pointer) {.cdecl, gcsafe, raises: [].} =
  # libdatachannel handles the SRTP decrypt/authenticate before this RTP callback.
  discard rtcSetMessageCallback(tr, onMediaRtp)

proc onState(pc:cint, state:cint, user:pointer) {.cdecl, gcsafe, raises: [].} =
  ensureForeignThread()
  if user == nil: return
  try:
    let names = ["new", "connecting", "connected", "disconnected", "failed", "closed"]
    let stateName = if state >= 0 and state < names.len.cint: names[state.int] else: "unknown"
    cast[NativeSession](user).socket.send($(%*{"type":"state", "state":stateName}))
  except CatchableError:
    discard

proc createNativeSession*(socket:WebSocket):NativeSession =
  var config: RtcConfig
  config.disableAutoNegotiation = true
  # No STUN server needed when a public candidate is available; if behind NAT,
  # provide a public host / TURN relay, or extend configuration with ICE URLs.
  let id = rtcCreatePeerConnection(addr config)
  if id < 0: return nil
  result = cast[NativeSession](allocShared0(sizeof(NativeSessionObj)))
  result.socket = socket
  result.pc = id
  # `pending` is a GC-managed seq; unlike the shared allocation itself,
  # it is only mutated from this WS worker's serialized callback.
  result.pending = @[]
  rtcSetUserPointer(id, cast[pointer](result))
  discard rtcSetLocalDescriptionCallback(id, sendLocalDescription)
  discard rtcSetLocalCandidateCallback(id, sendLocalCandidate)
  discard rtcSetDataChannelCallback(id, onDataChannel)
  discard rtcSetTrackCallback(id, onTrack)
  discard rtcSetStateChangeCallback(id, onState)

proc acceptOffer*(session:NativeSession, sdp:string):bool =
  if session == nil or sdp.len == 0: return false
  if rtcSetRemoteDescription(session.pc, sdp.cstring, "offer") != 0: return false
  session.remoteSet = true
  for item in session.pending:
    let mid = if item.mid.len == 0: nil else: item.mid.cstring
    discard rtcAddRemoteCandidate(session.pc, item.candidate.cstring, mid)
  session.pending.setLen(0)
  result = rtcSetLocalDescription(session.pc, "answer") == 0

proc acceptCandidate*(session:NativeSession, candidate, mid:string):bool =
  if session == nil: return false
  if candidate.len == 0: return true # end-of-candidates
  if not session.remoteSet:
    if session.pending.len >= 128: return false
    session.pending.add((candidate, mid))
    return true
  let midPtr = if mid.len == 0: nil else: mid.cstring
  result = rtcAddRemoteCandidate(session.pc, candidate.cstring, midPtr) == 0

proc destroyNativeSession*(session:NativeSession) =
  if session == nil: return
  # rtcDeletePeerConnection waits until the callbacks have completed.
  discard rtcDeletePeerConnection(session.pc)
  `=destroy`(session.pending)
  deallocShared(session)
