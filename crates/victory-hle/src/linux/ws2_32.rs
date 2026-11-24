//! Winsock (`ws2_32`) on Linux sockets, with the policy of
//! `modules::net` on Windows: a console without a network cable.
//!
//! * Sockets are real, so the title can talk to itself over loopback
//!   (its debug listener binds a TCP port). A `SOCKET` is the descriptor.
//! * Connects and datagram sends to anything but 127.0.0.0/8 and ::1 fail
//!   with `WSAENETUNREACH`; a bind to all interfaces is narrowed to
//!   loopback, a bind to any other address fails.
//! * No socket gets a local address on all interfaces, which is what Linux
//!   gives an unbound one: `listen` without a bind fails with `WSAEINVAL`,
//!   as it does on Windows, and a first datagram binds to loopback.
//! * Name lookups answer `localhost` and address literals only; the rest
//!   fail with `WSAHOST_NOT_FOUND`.
//!
//! Windows layouts are translated both ways: `sockaddr` families
//! (`AF_INET6` is 23), `fd_set` (a count and an array), `hostent` and
//! `addrinfo` (fields in another order), socket options and error codes.
//! Titles import most of this by ordinal (the Winsock 1.1 numbers).
#![allow(non_snake_case)]
#![allow(clippy::not_unsafe_ptr_arg_deref)]

use std::cell::RefCell;
use std::collections::HashSet;
use std::ffi::{CStr, CString, c_char, c_int, c_void};
use std::net::{IpAddr, Ipv4Addr, Ipv6Addr, SocketAddr};
use std::sync::Mutex;
use std::sync::atomic::{AtomicU32, Ordering};

use victory_loader::linux::teb;

use crate::{HleExport, HleModule, hle_export};

type Socket = usize;

const INVALID_SOCKET: Socket = !0;
const SOCKET_ERROR: i32 = -1;

const WSAEINTR: i32 = 10004;
const WSAEACCES: i32 = 10013;
const WSAEFAULT: i32 = 10014;
const WSAEINVAL: i32 = 10022;
const WSAEMFILE: i32 = 10024;
const WSAEWOULDBLOCK: i32 = 10035;
const WSAEALREADY: i32 = 10037;
const WSAENOTSOCK: i32 = 10038;
const WSAEDESTADDRREQ: i32 = 10039;
const WSAEMSGSIZE: i32 = 10040;
const WSAEPROTOTYPE: i32 = 10041;
const WSAENOPROTOOPT: i32 = 10042;
const WSAEPROTONOSUPPORT: i32 = 10043;
const WSAEOPNOTSUPP: i32 = 10045;
const WSAEAFNOSUPPORT: i32 = 10047;
const WSAEADDRINUSE: i32 = 10048;
const WSAEADDRNOTAVAIL: i32 = 10049;
const WSAENETDOWN: i32 = 10050;
const WSAENETUNREACH: i32 = 10051;
const WSAECONNABORTED: i32 = 10053;
const WSAECONNRESET: i32 = 10054;
const WSAENOBUFS: i32 = 10055;
const WSAEISCONN: i32 = 10056;
const WSAENOTCONN: i32 = 10057;
const WSAESHUTDOWN: i32 = 10058;
const WSAETIMEDOUT: i32 = 10060;
const WSAECONNREFUSED: i32 = 10061;
const WSAEHOSTUNREACH: i32 = 10065;
const WSAVERNOTSUPPORTED: i32 = 10092;
const WSANOTINITIALISED: i32 = 10093;
const WSATYPE_NOT_FOUND: i32 = 10109;
const WSAHOST_NOT_FOUND: i32 = 11001;

const AF_UNSPEC: i32 = 0;
const AF_INET: i32 = 2;
const AF_INET6: i32 = 23;

fn fail(code: i32) -> i32 {
    teb::set_last_error(code as u32);
    SOCKET_ERROR
}

/// The Winsock code for an errno.
fn wsa(errno: i32) -> i32 {
    match errno {
        libc::EINTR => WSAEINTR,
        libc::EBADF => WSAENOTSOCK,
        libc::EACCES | libc::EPERM => WSAEACCES,
        libc::EFAULT => WSAEFAULT,
        libc::EINVAL => WSAEINVAL,
        libc::EMFILE | libc::ENFILE => WSAEMFILE,
        libc::EAGAIN | libc::EINPROGRESS => WSAEWOULDBLOCK,
        libc::EALREADY => WSAEALREADY,
        libc::ENOTSOCK => WSAENOTSOCK,
        libc::EDESTADDRREQ => WSAEDESTADDRREQ,
        libc::EMSGSIZE => WSAEMSGSIZE,
        libc::EPROTOTYPE => WSAEPROTOTYPE,
        libc::ENOPROTOOPT => WSAENOPROTOOPT,
        libc::EPROTONOSUPPORT | libc::ESOCKTNOSUPPORT => WSAEPROTONOSUPPORT,
        libc::EOPNOTSUPP => WSAEOPNOTSUPP,
        libc::EAFNOSUPPORT => WSAEAFNOSUPPORT,
        libc::EADDRINUSE => WSAEADDRINUSE,
        libc::EADDRNOTAVAIL => WSAEADDRNOTAVAIL,
        libc::ENETDOWN => WSAENETDOWN,
        libc::ENETUNREACH => WSAENETUNREACH,
        libc::ECONNABORTED => WSAECONNABORTED,
        libc::ECONNRESET => WSAECONNRESET,
        libc::ENOBUFS | libc::ENOMEM => WSAENOBUFS,
        libc::EISCONN => WSAEISCONN,
        libc::ENOTCONN => WSAENOTCONN,
        libc::EPIPE | libc::ESHUTDOWN => WSAESHUTDOWN,
        libc::ETIMEDOUT => WSAETIMEDOUT,
        libc::ECONNREFUSED => WSAECONNREFUSED,
        libc::EHOSTUNREACH => WSAEHOSTUNREACH,
        _ => WSAEINVAL,
    }
}

fn errno() -> i32 {
    std::io::Error::last_os_error().raw_os_error().unwrap_or(0)
}

/// `SOCKET_ERROR` with the last OS error as the Winsock error.
fn os_fail() -> i32 {
    fail(wsa(errno()))
}

// ---- start-up and the socket set ----

static STARTED: AtomicU32 = AtomicU32::new(0);
static SOCKETS: Mutex<Option<HashSet<i32>>> = Mutex::new(None);

fn sockets<R>(f: impl FnOnce(&mut HashSet<i32>) -> R) -> R {
    let mut s = SOCKETS.lock().unwrap_or_else(|e| e.into_inner());
    f(s.get_or_insert_with(HashSet::new))
}

/// The descriptor of `s`, or the Winsock error for it.
fn fd(s: Socket) -> Result<i32, i32> {
    if STARTED.load(Ordering::Acquire) == 0 {
        return Err(WSANOTINITIALISED);
    }
    let fd = i32::try_from(s).map_err(|_| WSAENOTSOCK)?;
    if sockets(|set| set.contains(&fd)) { Ok(fd) } else { Err(WSAENOTSOCK) }
}

macro_rules! fd_or_fail {
    ($s:expr) => {
        match fd($s) {
            Ok(fd) => fd,
            Err(code) => return fail(code),
        }
    };
}

/// `WSADATA` (64-bit order).
#[repr(C)]
pub struct WsaData {
    version: u16,
    high_version: u16,
    max_sockets: u16,
    max_udp: u16,
    vendor: *const c_char,
    description: [u8; 257],
    status: [u8; 129],
}

pub extern "win64" fn WSAStartup(version: u16, data: *mut WsaData) -> i32 {
    if data.is_null() {
        return WSAEFAULT;
    }
    let (major, minor) = (version & 0xFF, version >> 8);
    if major < 1 || (major == 1 && minor < 1) {
        return WSAVERNOTSUPPORTED;
    }
    let mut d = WsaData {
        version: if major > 2 { 0x0202 } else { version },
        high_version: 0x0202,
        max_sockets: 0,
        max_udp: 0,
        vendor: std::ptr::null(),
        description: [0; 257],
        status: [0; 129],
    };
    d.description[..11].copy_from_slice(b"WinSock 2.0");
    d.status[..7].copy_from_slice(b"Running");
    // SAFETY: checked non-null; the title's WSADATA.
    unsafe { data.write(d) };
    STARTED.fetch_add(1, Ordering::AcqRel);
    0
}

pub extern "win64" fn WSACleanup() -> i32 {
    let ok = STARTED
        .fetch_update(Ordering::AcqRel, Ordering::Acquire, |n| n.checked_sub(1))
        .is_ok();
    if ok { 0 } else { fail(WSANOTINITIALISED) }
}

pub extern "win64" fn WSAGetLastError() -> i32 {
    teb::last_error() as i32
}

pub extern "win64" fn WSASetLastError(code: i32) {
    teb::set_last_error(code as u32);
}

// ---- addresses ----

/// A Linux `sockaddr` and its length.
struct HostAddr(libc::sockaddr_storage, libc::socklen_t);

/// The address a title's `sockaddr` names.
fn guest_addr(addr: *const u8, len: i32) -> Result<SocketAddr, i32> {
    let len = usize::try_from(len).map_err(|_| WSAEFAULT)?;
    if addr.is_null() || len < 4 {
        return Err(WSAEFAULT);
    }
    // SAFETY: the title passes `len` readable bytes.
    let b = unsafe { std::slice::from_raw_parts(addr, len) };
    let port = u16::from_be_bytes([b[2], b[3]]);
    match i32::from(u16::from_le_bytes([b[0], b[1]])) {
        AF_INET if len >= 16 => {
            let ip: [u8; 4] = b[4..8].try_into().expect("4 bytes");
            Ok((ip, port).into())
        }
        AF_INET6 if len >= 28 => {
            let ip: [u8; 16] = b[8..24].try_into().expect("16 bytes");
            let flow = u32::from_be_bytes(b[4..8].try_into().expect("4 bytes"));
            let scope = u32::from_le_bytes(b[24..28].try_into().expect("4 bytes"));
            Ok(SocketAddr::V6(std::net::SocketAddrV6::new(ip.into(), port, flow, scope)))
        }
        AF_INET | AF_INET6 => Err(WSAEFAULT),
        _ => Err(WSAEAFNOSUPPORT),
    }
}

fn host_addr(a: &SocketAddr) -> HostAddr {
    // SAFETY: all-zero is a valid sockaddr_storage.
    let mut s: libc::sockaddr_storage = unsafe { std::mem::zeroed() };
    let len = match a {
        SocketAddr::V4(v4) => {
            // SAFETY: sockaddr_storage is large and aligned enough.
            let sin = unsafe { &mut *(&mut s as *mut _ as *mut libc::sockaddr_in) };
            sin.sin_family = libc::AF_INET as u16;
            sin.sin_port = v4.port().to_be();
            sin.sin_addr.s_addr = u32::from_ne_bytes(v4.ip().octets());
            size_of::<libc::sockaddr_in>()
        }
        SocketAddr::V6(v6) => {
            // SAFETY: as above.
            let sin6 = unsafe { &mut *(&mut s as *mut _ as *mut libc::sockaddr_in6) };
            sin6.sin6_family = libc::AF_INET6 as u16;
            sin6.sin6_port = v6.port().to_be();
            sin6.sin6_flowinfo = v6.flowinfo().to_be();
            sin6.sin6_addr.s6_addr = v6.ip().octets();
            sin6.sin6_scope_id = v6.scope_id();
            size_of::<libc::sockaddr_in6>()
        }
    };
    HostAddr(s, len as libc::socklen_t)
}

fn from_host(s: &libc::sockaddr_storage) -> Option<SocketAddr> {
    match i32::from(s.ss_family) {
        libc::AF_INET => {
            // SAFETY: the family says this is a sockaddr_in.
            let sin = unsafe { &*(s as *const _ as *const libc::sockaddr_in) };
            Some((sin.sin_addr.s_addr.to_ne_bytes(), u16::from_be(sin.sin_port)).into())
        }
        libc::AF_INET6 => {
            // SAFETY: the family says this is a sockaddr_in6.
            let sin6 = unsafe { &*(s as *const _ as *const libc::sockaddr_in6) };
            Some(SocketAddr::V6(std::net::SocketAddrV6::new(
                sin6.sin6_addr.s6_addr.into(),
                u16::from_be(sin6.sin6_port),
                u32::from_be(sin6.sin6_flowinfo),
                sin6.sin6_scope_id,
            )))
        }
        _ => None,
    }
}

/// `a` in Windows layout.
fn guest_bytes(a: &SocketAddr) -> Vec<u8> {
    let mut b = Vec::with_capacity(28);
    match a {
        SocketAddr::V4(v4) => {
            b.extend((AF_INET as u16).to_le_bytes());
            b.extend(v4.port().to_be_bytes());
            b.extend(v4.ip().octets());
            b.extend([0; 8]);
        }
        SocketAddr::V6(v6) => {
            b.extend((AF_INET6 as u16).to_le_bytes());
            b.extend(v6.port().to_be_bytes());
            b.extend(v6.flowinfo().to_be_bytes());
            b.extend(v6.ip().octets());
            b.extend(v6.scope_id().to_le_bytes());
        }
    }
    b
}

/// Writes `a` to a title's `sockaddr` and length, as `accept` and
/// `getsockname` do; false (`WSAEFAULT`) if it does not fit.
fn put_guest_addr(a: Option<SocketAddr>, out: *mut u8, len: *mut i32) -> bool {
    if out.is_null() || len.is_null() {
        return true;
    }
    let bytes = a.map(|a| guest_bytes(&a)).unwrap_or_default();
    // SAFETY: the title's length and a buffer of that many bytes.
    unsafe {
        if (*len as usize) < bytes.len() || *len < 0 {
            return false;
        }
        std::ptr::copy_nonoverlapping(bytes.as_ptr(), out, bytes.len());
        *len = bytes.len() as i32;
    }
    true
}

fn loopback(ip: IpAddr) -> bool {
    ip.to_canonical().is_loopback()
}

fn once(what: &'static str, to: SocketAddr) {
    static SEEN: Mutex<Vec<(&str, SocketAddr)>> = Mutex::new(Vec::new());
    let mut seen = SEEN.lock().unwrap_or_else(|e| e.into_inner());
    if !seen.contains(&(what, to)) {
        tracing::info!(%to, "{what}");
        seen.push((what, to));
    }
}

/// Only loopback is reachable.
fn reachable(what: &'static str, to: SocketAddr) -> Result<(), i32> {
    if loopback(to.ip()) {
        return Ok(());
    }
    once(what, to);
    Err(WSAENETUNREACH)
}

// ---- sockets ----

pub extern "win64" fn socket(family: i32, kind: i32, protocol: i32) -> Socket {
    if STARTED.load(Ordering::Acquire) == 0 {
        fail(WSANOTINITIALISED);
        return INVALID_SOCKET;
    }
    let family = match family {
        AF_INET => libc::AF_INET,
        AF_INET6 => libc::AF_INET6,
        _ => {
            fail(WSAEAFNOSUPPORT);
            return INVALID_SOCKET;
        }
    };
    if !matches!(kind, libc::SOCK_STREAM | libc::SOCK_DGRAM) {
        fail(WSAEPROTONOSUPPORT);
        return INVALID_SOCKET;
    }
    // SAFETY: plain socket creation.
    let fd = unsafe { libc::socket(family, kind | libc::SOCK_CLOEXEC, protocol) };
    if fd < 0 {
        os_fail();
        return INVALID_SOCKET;
    }
    sockets(|s| s.insert(fd));
    fd as Socket
}

pub extern "win64" fn closesocket(s: Socket) -> i32 {
    let fd = fd_or_fail!(s);
    sockets(|set| set.remove(&fd));
    // SAFETY: a descriptor we opened, no longer in the set.
    if unsafe { libc::close(fd) } != 0 { os_fail() } else { 0 }
}

pub extern "win64" fn bind(s: Socket, addr: *const u8, len: i32) -> i32 {
    let fd = fd_or_fail!(s);
    let mut to = match guest_addr(addr, len) {
        Ok(a) => a,
        Err(code) => return fail(code),
    };
    // All interfaces means loopback: nothing on the network may connect in.
    if to.ip().is_unspecified() {
        to.set_ip(match to {
            SocketAddr::V4(_) => Ipv4Addr::LOCALHOST.into(),
            SocketAddr::V6(_) => Ipv6Addr::LOCALHOST.into(),
        });
    } else if !loopback(to.ip()) {
        return fail(WSAEADDRNOTAVAIL);
    }
    let a = host_addr(&to);
    // SAFETY: a valid sockaddr and its length.
    if unsafe { libc::bind(fd, &a.0 as *const _ as *const libc::sockaddr, a.1) } != 0 {
        return os_fail();
    }
    0
}

pub extern "win64" fn connect(s: Socket, addr: *const u8, len: i32) -> i32 {
    let fd = fd_or_fail!(s);
    let to = match guest_addr(addr, len).and_then(|a| reachable("connect refused: no network", a).map(|_| a)) {
        Ok(a) => a,
        Err(code) => return fail(code),
    };
    let a = host_addr(&to);
    // SAFETY: a valid sockaddr and its length.
    if unsafe { libc::connect(fd, &a.0 as *const _ as *const libc::sockaddr, a.1) } != 0 {
        return os_fail();
    }
    0
}

/// The local address of `fd`; unspecified with port 0 until it is bound.
fn local(fd: i32) -> Option<SocketAddr> {
    // SAFETY: all-zero is a valid sockaddr_storage.
    let mut a: libc::sockaddr_storage = unsafe { std::mem::zeroed() };
    let mut len = size_of::<libc::sockaddr_storage>() as libc::socklen_t;
    // SAFETY: room for any address.
    if unsafe { libc::getsockname(fd, &mut a as *mut _ as *mut libc::sockaddr, &mut len) } != 0 {
        return None;
    }
    from_host(&a)
}

/// Binds an unbound datagram socket to loopback before it sends to `to`.
/// Linux would bind it to all interfaces, where the network could answer.
fn bind_datagram(fd: i32, to: &SocketAddr) -> Result<(), i32> {
    let Some(mut from) = local(fd).filter(|a| a.port() == 0) else {
        return Ok(());
    };
    let mut kind: c_int = 0;
    let mut len = size_of::<c_int>() as libc::socklen_t;
    // SAFETY: an int and its size.
    let r = unsafe { libc::getsockopt(fd, libc::SOL_SOCKET, libc::SO_TYPE, &mut kind as *mut _ as *mut c_void, &mut len) };
    if r != 0 || kind != libc::SOCK_DGRAM {
        return Ok(());
    }
    from.set_ip(match (from, to.ip().to_canonical()) {
        (SocketAddr::V4(_), _) => Ipv4Addr::LOCALHOST.into(),
        // A dual-stack socket sending to 127.0.0.1 needs the mapped address.
        (SocketAddr::V6(_), IpAddr::V4(_)) => Ipv4Addr::LOCALHOST.to_ipv6_mapped().into(),
        (SocketAddr::V6(_), IpAddr::V6(_)) => Ipv6Addr::LOCALHOST.into(),
    });
    let a = host_addr(&from);
    // SAFETY: a valid sockaddr and its length.
    if unsafe { libc::bind(fd, &a.0 as *const _ as *const libc::sockaddr, a.1) } != 0 {
        return Err(wsa(errno()));
    }
    Ok(())
}

pub extern "win64" fn listen(s: Socket, backlog: i32) -> i32 {
    let fd = fd_or_fail!(s);
    // Linux binds an unbound socket to all interfaces here. Winsock refuses
    // it, and `bind` only ever gives a loopback address.
    if !local(fd).is_some_and(|a| loopback(a.ip())) {
        return fail(WSAEINVAL);
    }
    // SAFETY: plain call on our descriptor.
    if unsafe { libc::listen(fd, backlog) } != 0 { os_fail() } else { 0 }
}

pub extern "win64" fn accept(s: Socket, addr: *mut u8, len: *mut i32) -> Socket {
    let fd = match fd(s) {
        Ok(fd) => fd,
        Err(code) => {
            fail(code);
            return INVALID_SOCKET;
        }
    };
    // SAFETY: all-zero is a valid sockaddr_storage.
    let mut peer: libc::sockaddr_storage = unsafe { std::mem::zeroed() };
    let mut peer_len = size_of::<libc::sockaddr_storage>() as libc::socklen_t;
    // SAFETY: room for any address.
    let new = unsafe {
        libc::accept4(fd, &mut peer as *mut _ as *mut libc::sockaddr, &mut peer_len, libc::SOCK_CLOEXEC)
    };
    if new < 0 {
        os_fail();
        return INVALID_SOCKET;
    }
    if !put_guest_addr(from_host(&peer), addr, len) {
        // SAFETY: the descriptor accept just made.
        unsafe { libc::close(new) };
        fail(WSAEFAULT);
        return INVALID_SOCKET;
    }
    sockets(|set| set.insert(new));
    new as Socket
}

fn name(s: Socket, addr: *mut u8, len: *mut i32, peer: bool) -> i32 {
    let fd = fd_or_fail!(s);
    if addr.is_null() || len.is_null() {
        return fail(WSAEFAULT);
    }
    // SAFETY: all-zero is a valid sockaddr_storage.
    let mut a: libc::sockaddr_storage = unsafe { std::mem::zeroed() };
    let mut a_len = size_of::<libc::sockaddr_storage>() as libc::socklen_t;
    let p = &mut a as *mut _ as *mut libc::sockaddr;
    // SAFETY: room for any address.
    let r = unsafe {
        if peer { libc::getpeername(fd, p, &mut a_len) } else { libc::getsockname(fd, p, &mut a_len) }
    };
    if r != 0 {
        return os_fail();
    }
    if put_guest_addr(from_host(&a), addr, len) { 0 } else { fail(WSAEFAULT) }
}

pub extern "win64" fn getsockname(s: Socket, addr: *mut u8, len: *mut i32) -> i32 {
    name(s, addr, len, false)
}

pub extern "win64" fn getpeername(s: Socket, addr: *mut u8, len: *mut i32) -> i32 {
    name(s, addr, len, true)
}

const MSG_WAITALL: i32 = 8;

/// Windows `MSG_*` flags as Linux ones (`MSG_OOB`, `MSG_PEEK` and
/// `MSG_DONTROUTE` agree).
fn flags(flags: i32) -> i32 {
    (flags & 7) | if flags & MSG_WAITALL != 0 { libc::MSG_WAITALL } else { 0 }
}

fn result(n: isize) -> i32 {
    if n < 0 { os_fail() } else { n as i32 }
}

pub extern "win64" fn send(s: Socket, data: *const u8, len: i32, f: i32) -> i32 {
    let fd = fd_or_fail!(s);
    // SAFETY: the title passes `len` readable bytes.
    result(unsafe { libc::send(fd, data.cast(), len.max(0) as usize, flags(f) | libc::MSG_NOSIGNAL) })
}

pub extern "win64" fn sendto(s: Socket, data: *const u8, len: i32, f: i32, to: *const u8, to_len: i32) -> i32 {
    let fd = fd_or_fail!(s);
    if to.is_null() {
        return send(s, data, len, f);
    }
    let a = match guest_addr(to, to_len)
        .and_then(|a| reachable("datagram dropped: no network", a).map(|_| a))
        .and_then(|a| bind_datagram(fd, &a).map(|_| a))
    {
        Ok(a) => host_addr(&a),
        Err(code) => return fail(code),
    };
    // SAFETY: `len` readable bytes and a valid sockaddr.
    result(unsafe {
        libc::sendto(
            fd,
            data.cast(),
            len.max(0) as usize,
            flags(f) | libc::MSG_NOSIGNAL,
            &a.0 as *const _ as *const libc::sockaddr,
            a.1,
        )
    })
}

pub extern "win64" fn recv(s: Socket, data: *mut u8, len: i32, f: i32) -> i32 {
    let fd = fd_or_fail!(s);
    // SAFETY: the title passes `len` writable bytes.
    result(unsafe { libc::recv(fd, data.cast(), len.max(0) as usize, flags(f)) })
}

pub extern "win64" fn recvfrom(s: Socket, data: *mut u8, len: i32, f: i32, from: *mut u8, from_len: *mut i32) -> i32 {
    let fd = fd_or_fail!(s);
    // SAFETY: all-zero is a valid sockaddr_storage.
    let mut a: libc::sockaddr_storage = unsafe { std::mem::zeroed() };
    let mut a_len = size_of::<libc::sockaddr_storage>() as libc::socklen_t;
    // SAFETY: `len` writable bytes and room for any address.
    let n = unsafe {
        libc::recvfrom(fd, data.cast(), len.max(0) as usize, flags(f), &mut a as *mut _ as *mut libc::sockaddr, &mut a_len)
    };
    if n < 0 {
        return os_fail();
    }
    if !put_guest_addr(from_host(&a), from, from_len) {
        return fail(WSAEFAULT);
    }
    n as i32
}

pub extern "win64" fn shutdown(s: Socket, how: i32) -> i32 {
    let fd = fd_or_fail!(s);
    // SD_RECEIVE, SD_SEND and SD_BOTH are SHUT_RD, SHUT_WR and SHUT_RDWR.
    // SAFETY: plain call on our descriptor.
    if unsafe { libc::shutdown(fd, how) } != 0 { os_fail() } else { 0 }
}

const FIONBIO: u32 = 0x8004_667E;
const FIONREAD: u32 = 0x4004_667F;
const SIOCATMARK: u32 = 0x4004_7307;
const LINUX_SIOCATMARK: libc::Ioctl = 0x8905;

pub extern "win64" fn ioctlsocket(s: Socket, command: u32, arg: *mut u32) -> i32 {
    let fd = fd_or_fail!(s);
    if arg.is_null() {
        return fail(WSAEFAULT);
    }
    // SAFETY: plain calls on our descriptor; `arg` is the title's u_long.
    unsafe {
        match command {
            FIONBIO => {
                let old = libc::fcntl(fd, libc::F_GETFL);
                let new = if *arg != 0 { old | libc::O_NONBLOCK } else { old & !libc::O_NONBLOCK };
                if old < 0 || libc::fcntl(fd, libc::F_SETFL, new) < 0 {
                    return os_fail();
                }
            }
            FIONREAD | SIOCATMARK => {
                let mut n: c_int = 0;
                let request = if command == FIONREAD { libc::FIONREAD } else { LINUX_SIOCATMARK };
                if libc::ioctl(fd, request, &mut n) < 0 {
                    return os_fail();
                }
                *arg = n as u32;
            }
            _ => return fail(WSAEINVAL),
        }
    }
    0
}

// ---- options ----

const SOL_SOCKET: i32 = 0xFFFF;
const IPPROTO_IP: i32 = 0;
const IPPROTO_TCP: i32 = 6;
const IPPROTO_IPV6: i32 = 41;
const SO_DONTLINGER: i32 = !0x80;
const SO_EXCLUSIVEADDRUSE: i32 = !4;
const SO_LINGER: i32 = 0x80;
const SO_SNDTIMEO: i32 = 0x1005;
const SO_RCVTIMEO: i32 = 0x1006;
const SO_ERROR: i32 = 0x1007;

/// The Linux level and name of a Windows option with an `int` value.
fn int_option(level: i32, name: i32) -> Option<(i32, i32)> {
    let linux = match (level, name) {
        (SOL_SOCKET, 4) => libc::SO_REUSEADDR,
        (SOL_SOCKET, 8) => libc::SO_KEEPALIVE,
        (SOL_SOCKET, 0x20) => libc::SO_BROADCAST,
        (SOL_SOCKET, 0x100) => libc::SO_OOBINLINE,
        (SOL_SOCKET, 0x1001) => libc::SO_SNDBUF,
        (SOL_SOCKET, 0x1002) => libc::SO_RCVBUF,
        (SOL_SOCKET, 0x1008) => libc::SO_TYPE,
        (SOL_SOCKET, 2) => libc::SO_ACCEPTCONN,
        (IPPROTO_TCP, 1) => return Some((libc::IPPROTO_TCP, libc::TCP_NODELAY)),
        (IPPROTO_IP, 3) => return Some((libc::IPPROTO_IP, libc::IP_TOS)),
        (IPPROTO_IP, 4) => return Some((libc::IPPROTO_IP, libc::IP_TTL)),
        (IPPROTO_IP, 10) => return Some((libc::IPPROTO_IP, libc::IP_MULTICAST_TTL)),
        (IPPROTO_IP, 11) => return Some((libc::IPPROTO_IP, libc::IP_MULTICAST_LOOP)),
        (IPPROTO_IPV6, 27) => return Some((libc::IPPROTO_IPV6, libc::IPV6_V6ONLY)),
        _ => return None,
    };
    Some((libc::SOL_SOCKET, linux))
}

/// Reads an option value of up to four bytes (Windows takes a one-byte
/// `BOOL` for some).
fn read_int(value: *const u8, len: i32) -> Result<i32, i32> {
    if value.is_null() || len < 1 {
        return Err(WSAEFAULT);
    }
    let mut b = [0u8; 4];
    // SAFETY: the title passes `len` readable bytes.
    unsafe { std::ptr::copy_nonoverlapping(value, b.as_mut_ptr(), (len as usize).min(4)) };
    Ok(i32::from_le_bytes(b))
}

fn set_raw<T>(fd: i32, level: i32, name: i32, v: &T) -> i32 {
    // SAFETY: a value of the option's type and its size.
    let r = unsafe { libc::setsockopt(fd, level, name, v as *const T as *const c_void, size_of::<T>() as u32) };
    if r != 0 { os_fail() } else { 0 }
}

pub extern "win64" fn setsockopt(s: Socket, level: i32, name: i32, value: *const u8, len: i32) -> i32 {
    let fd = fd_or_fail!(s);
    let v = match read_int(value, len) {
        Ok(v) => v,
        Err(code) => return fail(code),
    };
    match (level, name) {
        (SOL_SOCKET, SO_EXCLUSIVEADDRUSE) => 0,
        (SOL_SOCKET, SO_DONTLINGER) => {
            let l = libc::linger { l_onoff: (v == 0) as c_int, l_linger: 0 };
            set_raw(fd, libc::SOL_SOCKET, libc::SO_LINGER, &l)
        }
        (SOL_SOCKET, SO_LINGER) => {
            // Windows' linger is two u_shorts.
            let l = libc::linger {
                l_onoff: (v & 0xFFFF) as c_int,
                l_linger: (v >> 16) & 0xFFFF,
            };
            set_raw(fd, libc::SOL_SOCKET, libc::SO_LINGER, &l)
        }
        (SOL_SOCKET, SO_SNDTIMEO | SO_RCVTIMEO) => {
            // Milliseconds on Windows, a timeval here.
            let t = libc::timeval {
                tv_sec: (v as u32 / 1000).into(),
                tv_usec: (v as u32 % 1000 * 1000).into(),
            };
            let name = if name == SO_SNDTIMEO { libc::SO_SNDTIMEO } else { libc::SO_RCVTIMEO };
            set_raw(fd, libc::SOL_SOCKET, name, &t)
        }
        _ => match int_option(level, name) {
            Some((l, n)) => set_raw(fd, l, n, &v),
            None => fail(WSAENOPROTOOPT),
        },
    }
}

pub extern "win64" fn getsockopt(s: Socket, level: i32, name: i32, value: *mut u8, len: *mut i32) -> i32 {
    let fd = fd_or_fail!(s);
    // SAFETY: the title's length, read when it is not null.
    if value.is_null() || len.is_null() || unsafe { *len } < 4 {
        return fail(WSAEFAULT);
    }
    let get = |l: i32, n: i32| -> Result<i32, i32> {
        let mut v: c_int = 0;
        let mut size = size_of::<c_int>() as u32;
        // SAFETY: an int-sized value.
        let r = unsafe { libc::getsockopt(fd, l, n, &mut v as *mut c_int as *mut c_void, &mut size) };
        if r != 0 { Err(wsa(errno())) } else { Ok(v) }
    };
    let v = match (level, name) {
        (SOL_SOCKET, SO_ERROR) => get(libc::SOL_SOCKET, libc::SO_ERROR).map(|e| if e == 0 { 0 } else { wsa(e) }),
        (SOL_SOCKET, SO_EXCLUSIVEADDRUSE) => Ok(0),
        (SOL_SOCKET, SO_SNDTIMEO | SO_RCVTIMEO) => {
            let mut t = libc::timeval { tv_sec: 0, tv_usec: 0 };
            let mut size = size_of::<libc::timeval>() as u32;
            let n = if name == SO_SNDTIMEO { libc::SO_SNDTIMEO } else { libc::SO_RCVTIMEO };
            // SAFETY: a timeval-sized value.
            let r = unsafe { libc::getsockopt(fd, libc::SOL_SOCKET, n, &mut t as *mut _ as *mut c_void, &mut size) };
            if r != 0 { Err(wsa(errno())) } else { Ok((t.tv_sec * 1000 + t.tv_usec / 1000) as i32) }
        }
        _ => match int_option(level, name) {
            Some((l, n)) => get(l, n),
            None => Err(WSAENOPROTOOPT),
        },
    };
    match v {
        Ok(v) => {
            // SAFETY: checked: at least four writable bytes.
            unsafe {
                value.cast::<i32>().write_unaligned(v);
                *len = 4;
            }
            0
        }
        Err(code) => fail(code),
    }
}

// ---- select ----

/// Windows `fd_set`: a count, then that many `SOCKET`s.
#[repr(C)]
pub struct FdSet {
    count: u32,
    sockets: [Socket; 64],
}

fn set_sockets<'a>(set: *mut FdSet) -> &'a mut [Socket] {
    if set.is_null() {
        return &mut [];
    }
    // SAFETY: the title's fd_set holds `count` sockets (titles may make the
    // array larger than 64; the count says how many there are).
    unsafe {
        let count = (*set).count as usize;
        std::slice::from_raw_parts_mut((*set).sockets.as_mut_ptr(), count)
    }
}

pub extern "win64" fn __WSAFDIsSet(s: Socket, set: *mut FdSet) -> i32 {
    set_sockets(set).contains(&s) as i32
}

/// `timeval` with 32-bit fields, as Windows has it.
#[repr(C)]
pub struct Timeval {
    sec: i32,
    usec: i32,
}

pub extern "win64" fn select(_nfds: i32, read: *mut FdSet, write: *mut FdSet, except: *mut FdSet, timeout: *const Timeval) -> i32 {
    let sets = [read, write, except];
    let mut polls: Vec<libc::pollfd> = Vec::new();
    let mut slot: Vec<(usize, usize)> = Vec::new(); // (set, index) per poll entry
    for (k, &set) in sets.iter().enumerate() {
        for (i, &s) in set_sockets(set).iter().enumerate() {
            let fd = match fd(s) {
                Ok(fd) => fd,
                Err(code) => return fail(code),
            };
            let events = [libc::POLLIN, libc::POLLOUT, libc::POLLPRI][k];
            polls.push(libc::pollfd { fd, events, revents: 0 });
            slot.push((k, i));
        }
    }
    if polls.is_empty() && timeout.is_null() {
        return fail(WSAEINVAL);
    }
    let ms = if timeout.is_null() {
        -1
    } else {
        // SAFETY: checked non-null.
        let t = unsafe { &*timeout };
        (i64::from(t.sec) * 1000 + (i64::from(t.usec.max(0)) + 999) / 1000).clamp(0, i64::from(i32::MAX)) as i32
    };
    // SAFETY: `polls.len()` entries.
    let n = unsafe { libc::poll(polls.as_mut_ptr(), polls.len() as libc::nfds_t, ms) };
    if n < 0 {
        return os_fail();
    }
    // Each set keeps only its ready sockets, in order.
    let mut kept: [Vec<Socket>; 3] = Default::default();
    for (p, &(k, i)) in polls.iter().zip(&slot) {
        let r = p.revents;
        let ready = match k {
            0 => r & (libc::POLLIN | libc::POLLHUP | libc::POLLERR) != 0,
            1 => r & libc::POLLOUT != 0 && r & libc::POLLERR == 0,
            _ => r & (libc::POLLPRI | libc::POLLERR) != 0,
        };
        if ready {
            kept[k].push(set_sockets(sets[k])[i]);
        }
    }
    let mut total = 0;
    for (k, &set) in sets.iter().enumerate() {
        if set.is_null() {
            continue;
        }
        // SAFETY: the title's fd_set; we write back no more than it held.
        unsafe {
            let out = (*set).sockets.as_mut_ptr();
            std::ptr::copy_nonoverlapping(kept[k].as_ptr(), out, kept[k].len());
            (*set).count = kept[k].len() as u32;
        }
        total += kept[k].len() as i32;
    }
    total
}

// ---- names ----

pub extern "win64" fn htons(x: u16) -> u16 {
    x.to_be()
}

pub extern "win64" fn ntohs(x: u16) -> u16 {
    u16::from_be(x)
}

pub extern "win64" fn htonl(x: u32) -> u32 {
    x.to_be()
}

pub extern "win64" fn ntohl(x: u32) -> u32 {
    u32::from_be(x)
}

fn narrow(text: *const c_char) -> Option<String> {
    // SAFETY: the title passes a NUL-terminated string or null.
    (!text.is_null()).then(|| unsafe { CStr::from_ptr(text) }.to_string_lossy().into_owned())
}

/// `INADDR_NONE` for anything but a dotted IPv4 address.
pub extern "win64" fn inet_addr(text: *const c_char) -> u32 {
    narrow(text)
        .and_then(|t| t.trim().parse::<Ipv4Addr>().ok())
        .map_or(u32::MAX, |ip| u32::from_ne_bytes(ip.octets()))
}

pub extern "win64" fn gethostname(out: *mut c_char, len: i32) -> i32 {
    if out.is_null() || len <= 0 {
        return fail(WSAEFAULT);
    }
    let mut name = [0u8; 256];
    // SAFETY: room for 256 bytes; Linux NUL-terminates names that fit.
    if unsafe { libc::gethostname(name.as_mut_ptr().cast(), name.len() - 1) } != 0 {
        return os_fail();
    }
    let n = name.iter().position(|&b| b == 0).unwrap_or(0);
    if n + 1 > len as usize {
        return fail(WSAEFAULT);
    }
    // SAFETY: checked: `len` bytes hold the name and its NUL.
    unsafe { std::ptr::copy_nonoverlapping(name.as_ptr(), out.cast(), n + 1) };
    0
}

/// What an offline machine can look up: `localhost` and literals.
fn lookup(name: &str) -> Option<Vec<IpAddr>> {
    if name.is_empty() || name.eq_ignore_ascii_case("localhost") {
        return Some(vec![Ipv4Addr::LOCALHOST.into(), Ipv6Addr::LOCALHOST.into()]);
    }
    if let Ok(ip) = name.trim_matches(['[', ']']).parse::<IpAddr>() {
        return Some(vec![ip]);
    }
    tracing::info!(name, "name lookup failed: no network");
    None
}

/// `hostent` (64-bit layout); its storage is per thread, as on Windows.
#[repr(C)]
pub struct Hostent {
    name: *const c_char,
    aliases: *const *const c_char,
    kind: i16,
    length: i16,
    addresses: *const *const u8,
}

struct HostentStore {
    entry: Hostent,
    name: CString,
    aliases: [*const c_char; 1],
    address: [u8; 4],
    list: [*const u8; 2],
}

thread_local! {
    static HOSTENT: RefCell<Option<Box<HostentStore>>> = const { RefCell::new(None) };
}

pub extern "win64" fn gethostbyname(name: *const c_char) -> *const Hostent {
    let text = narrow(name).unwrap_or_default();
    let Some(ip) = lookup(&text).and_then(|ips| {
        ips.into_iter().find_map(|ip| match ip {
            IpAddr::V4(v4) => Some(v4),
            IpAddr::V6(_) => None,
        })
    }) else {
        fail(WSAHOST_NOT_FOUND);
        return std::ptr::null();
    };
    let mut store = Box::new(HostentStore {
        entry: Hostent {
            name: std::ptr::null(),
            aliases: std::ptr::null(),
            kind: AF_INET as i16,
            length: 4,
            addresses: std::ptr::null(),
        },
        name: CString::new(if text.is_empty() { "localhost".into() } else { text }).unwrap_or_default(),
        aliases: [std::ptr::null()],
        address: ip.octets(),
        list: [std::ptr::null(); 2],
    });
    store.list[0] = store.address.as_ptr();
    store.entry.name = store.name.as_ptr();
    store.entry.aliases = store.aliases.as_ptr();
    store.entry.addresses = store.list.as_ptr();
    let p: *const Hostent = &store.entry;
    HOSTENT.with(|h| *h.borrow_mut() = Some(store));
    p
}

/// `ADDRINFOA`.
#[repr(C)]
pub struct AddrInfo {
    flags: i32,
    family: i32,
    kind: i32,
    protocol: i32,
    addr_len: usize,
    canonical: *mut c_char,
    addr: *mut u8,
    next: *mut AddrInfo,
}

/// One result: the `ADDRINFOA` first, so the title's pointer frees it.
#[repr(C)]
struct Entry {
    info: AddrInfo,
    addr: [u8; 28],
    canonical: Option<CString>,
}

const AI_PASSIVE: i32 = 1;
const AI_CANONNAME: i32 = 2;
const AI_NUMERICHOST: i32 = 4;

pub extern "win64" fn getaddrinfo(node: *const c_char, service: *const c_char, hints: *const AddrInfo, out: *mut *mut AddrInfo) -> i32 {
    if out.is_null() || (node.is_null() && service.is_null()) {
        return WSAHOST_NOT_FOUND;
    }
    // SAFETY: the title's hints, or null.
    let (flags, family, kind, protocol) = match unsafe { hints.as_ref() } {
        Some(h) => (h.flags, h.family, h.kind, h.protocol),
        None => (0, AF_UNSPEC, 0, 0),
    };
    if !matches!(family, AF_UNSPEC | AF_INET | AF_INET6) {
        return WSAEAFNOSUPPORT;
    }
    let port = match narrow(service) {
        None => 0,
        Some(s) => match s.parse::<u16>() {
            Ok(p) => p,
            Err(_) => match s.as_str() {
                "http" => 80,
                "https" => 443,
                _ => return WSATYPE_NOT_FOUND,
            },
        },
    };
    let node_name = narrow(node);
    let ips = match &node_name {
        None if flags & AI_PASSIVE != 0 => vec![Ipv4Addr::UNSPECIFIED.into(), Ipv6Addr::UNSPECIFIED.into()],
        None => vec![Ipv4Addr::LOCALHOST.into(), Ipv6Addr::LOCALHOST.into()],
        Some(n) if flags & AI_NUMERICHOST != 0 && n.parse::<IpAddr>().is_err() => return WSAHOST_NOT_FOUND,
        Some(n) => match lookup(n) {
            Some(ips) => ips,
            None => return WSAHOST_NOT_FOUND,
        },
    };
    let wanted = |ip: &IpAddr| match family {
        AF_INET => ip.is_ipv4(),
        AF_INET6 => ip.is_ipv6(),
        _ => true,
    };
    let mut head: *mut AddrInfo = std::ptr::null_mut();
    for ip in ips.into_iter().filter(wanted).rev() {
        let bytes = guest_bytes(&SocketAddr::new(ip, port));
        let mut e = Box::new(Entry {
            info: AddrInfo {
                flags: 0,
                family: if ip.is_ipv4() { AF_INET } else { AF_INET6 },
                kind,
                protocol,
                addr_len: bytes.len(),
                canonical: std::ptr::null_mut(),
                addr: std::ptr::null_mut(),
                next: head,
            },
            addr: [0; 28],
            canonical: None,
        });
        e.addr[..bytes.len()].copy_from_slice(&bytes);
        e.info.addr = e.addr.as_mut_ptr();
        head = Box::into_raw(e).cast();
    }
    if head.is_null() {
        return WSAHOST_NOT_FOUND;
    }
    if flags & AI_CANONNAME != 0 {
        // SAFETY: the first entry we just made.
        let first = unsafe { &mut *(head as *mut Entry) };
        let name = CString::new(node_name.unwrap_or_else(|| "localhost".into())).unwrap_or_default();
        first.info.canonical = name.as_ptr() as *mut c_char;
        first.canonical = Some(name);
    }
    // SAFETY: checked non-null.
    unsafe { *out = head };
    0
}

pub extern "win64" fn freeaddrinfo(mut info: *mut AddrInfo) {
    while !info.is_null() {
        // SAFETY: an entry from `getaddrinfo`, freed once.
        let e = unsafe { Box::from_raw(info as *mut Entry) };
        info = e.info.next;
    }
}

/// Winsock imports by ordinal; the Winsock 1.1 numbers are fixed.
fn ordinal(name: &'static str, f: *const ()) -> HleExport {
    HleExport { name, address: f as usize }
}

pub(crate) fn module() -> HleModule {
    HleModule {
        name: "ws2_32",
        dlls: &["ws2_32.dll"],
        exports: vec![
            hle_export!(WSAStartup),
            hle_export!(WSACleanup),
            hle_export!(WSAGetLastError),
            hle_export!(WSASetLastError),
            hle_export!(__WSAFDIsSet),
            hle_export!(accept),
            hle_export!(bind),
            hle_export!(closesocket),
            hle_export!(connect),
            hle_export!(getpeername),
            hle_export!(getsockname),
            hle_export!(getsockopt),
            hle_export!(htonl),
            hle_export!(htons),
            hle_export!(ioctlsocket),
            hle_export!(inet_addr),
            hle_export!(listen),
            hle_export!(ntohl),
            hle_export!(ntohs),
            hle_export!(recv),
            hle_export!(recvfrom),
            hle_export!(select),
            hle_export!(send),
            hle_export!(sendto),
            hle_export!(setsockopt),
            hle_export!(shutdown),
            hle_export!(socket),
            hle_export!(gethostbyname),
            hle_export!(gethostname),
            hle_export!(getaddrinfo),
            hle_export!(freeaddrinfo),
            ordinal("#1", accept as *const ()),
            ordinal("#2", bind as *const ()),
            ordinal("#3", closesocket as *const ()),
            ordinal("#4", connect as *const ()),
            ordinal("#5", getpeername as *const ()),
            ordinal("#6", getsockname as *const ()),
            ordinal("#7", getsockopt as *const ()),
            ordinal("#8", htonl as *const ()),
            ordinal("#9", htons as *const ()),
            ordinal("#10", ioctlsocket as *const ()),
            ordinal("#11", inet_addr as *const ()),
            ordinal("#13", listen as *const ()),
            ordinal("#14", ntohl as *const ()),
            ordinal("#15", ntohs as *const ()),
            ordinal("#16", recv as *const ()),
            ordinal("#17", recvfrom as *const ()),
            ordinal("#18", select as *const ()),
            ordinal("#19", send as *const ()),
            ordinal("#20", sendto as *const ()),
            ordinal("#21", setsockopt as *const ()),
            ordinal("#22", shutdown as *const ()),
            ordinal("#23", socket as *const ()),
            ordinal("#52", gethostbyname as *const ()),
            ordinal("#57", gethostname as *const ()),
            ordinal("#111", WSAGetLastError as *const ()),
            ordinal("#112", WSASetLastError as *const ()),
            ordinal("#115", WSAStartup as *const ()),
            ordinal("#116", WSACleanup as *const ()),
            ordinal("#151", __WSAFDIsSet as *const ()),
        ],
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn v4(ip: [u8; 4], port: u16) -> [u8; 16] {
        let mut a = [0u8; 16];
        a[..2].copy_from_slice(&(AF_INET as u16).to_le_bytes());
        a[2..4].copy_from_slice(&port.to_be_bytes());
        a[4..8].copy_from_slice(&ip);
        a
    }

    fn start() {
        // SAFETY: an all-zero WSADATA is valid to overwrite.
        let mut data: WsaData = unsafe { std::mem::zeroed() };
        assert_eq!(WSAStartup(0x0202, &mut data), 0);
        assert_eq!(data.high_version, 0x0202);
    }

    #[test]
    fn layouts_are_windows() {
        assert_eq!(size_of::<WsaData>(), 408);
        assert_eq!(size_of::<AddrInfo>(), 48);
        assert_eq!(std::mem::offset_of!(AddrInfo, addr), 32);
        assert_eq!(std::mem::offset_of!(Hostent, addresses), 24);
        assert_eq!(std::mem::offset_of!(FdSet, sockets), 8);
    }

    #[test]
    fn calls_before_start_up_and_on_non_sockets_fail() {
        start();
        assert_eq!(closesocket(12345), SOCKET_ERROR);
        assert_eq!(WSAGetLastError(), WSAENOTSOCK);
        assert_eq!(socket(99, 1, 0), INVALID_SOCKET);
        assert_eq!(WSAGetLastError(), WSAEAFNOSUPPORT);
        assert_eq!(WSAStartup(0x0001, std::ptr::null_mut()), WSAEFAULT);
    }

    #[test]
    fn only_loopback_is_reachable() {
        start();
        let s = socket(AF_INET, libc::SOCK_STREAM, 6);
        assert_ne!(s, INVALID_SOCKET);
        // TEST-NET-1: never routed, and never tried here.
        let away = v4([192, 0, 2, 1], 80);
        assert_eq!(connect(s, away.as_ptr(), 16), SOCKET_ERROR);
        assert_eq!(WSAGetLastError(), WSAENETUNREACH);
        assert_eq!(bind(s, away.as_ptr(), 16), SOCKET_ERROR);
        assert_eq!(WSAGetLastError(), WSAEADDRNOTAVAIL);
        let u = socket(AF_INET, libc::SOCK_DGRAM, 17);
        assert_eq!(sendto(u, b"x".as_ptr(), 1, 0, away.as_ptr(), 16), SOCKET_ERROR);
        assert_eq!(WSAGetLastError(), WSAENETUNREACH);
        assert_eq!(closesocket(u), 0);
        assert_eq!(closesocket(s), 0);
    }

    #[test]
    fn a_listener_on_any_address_serves_loopback() {
        start();
        let server = socket(AF_INET, libc::SOCK_STREAM, 6);
        let one = 1i32.to_le_bytes();
        assert_eq!(setsockopt(server, SOL_SOCKET, 4, one.as_ptr(), 4), 0);
        assert_eq!(bind(server, v4([0; 4], 0).as_ptr(), 16), 0);
        assert_eq!(listen(server, 1), 0);
        let mut addr = [0u8; 16];
        let mut len = 16;
        assert_eq!(getsockname(server, addr.as_mut_ptr(), &mut len), 0);
        assert_eq!((len, &addr[..2], &addr[4..8]), (16, &[2u8, 0][..], &[127u8, 0, 0, 1][..]));

        let client = socket(AF_INET, libc::SOCK_STREAM, 6);
        assert_eq!(connect(client, addr.as_ptr(), 16), 0);
        let mut set = FdSet { count: 1, sockets: [0; 64] };
        set.sockets[0] = server;
        let t = Timeval { sec: 2, usec: 0 };
        assert_eq!(select(0, &mut set, std::ptr::null_mut(), std::ptr::null_mut(), &t), 1);
        assert_eq!(__WSAFDIsSet(server, &mut set), 1);
        let mut peer = [0u8; 16];
        let mut peer_len = 16;
        let conn = accept(server, peer.as_mut_ptr(), &mut peer_len);
        assert_ne!(conn, INVALID_SOCKET);
        assert_eq!(send(client, b"hi".as_ptr(), 2, 0), 2);
        let mut got = [0u8; 4];
        assert_eq!(recv(conn, got.as_mut_ptr(), 4, 0), 2);
        assert_eq!(&got[..2], b"hi");

        // Non-blocking with nothing to read: WSAEWOULDBLOCK.
        let mut on = 1u32;
        assert_eq!(ioctlsocket(conn, FIONBIO, &mut on), 0);
        assert_eq!(recv(conn, got.as_mut_ptr(), 4, 0), SOCKET_ERROR);
        assert_eq!(WSAGetLastError(), WSAEWOULDBLOCK);
        let mut v = [0u8; 4];
        let mut vlen = 4;
        assert_eq!(getsockopt(conn, SOL_SOCKET, SO_ERROR, v.as_mut_ptr(), &mut vlen), 0);
        assert_eq!(i32::from_le_bytes(v), 0);
        assert_eq!(getsockopt(conn, SOL_SOCKET, 0x7777, v.as_mut_ptr(), &mut vlen), SOCKET_ERROR);
        assert_eq!(WSAGetLastError(), WSAENOPROTOOPT);
        for s in [conn, client, server] {
            assert_eq!(closesocket(s), 0);
        }
    }

    #[test]
    fn an_unbound_socket_never_gets_all_interfaces() {
        start();
        let local_of = |s: Socket| {
            let mut addr = [0u8; 16];
            let mut len = 16;
            assert_eq!(getsockname(s, addr.as_mut_ptr(), &mut len), 0);
            ([addr[4], addr[5], addr[6], addr[7]], u16::from_be_bytes([addr[2], addr[3]]))
        };

        // Linux would listen on 0.0.0.0 and a port of its choosing.
        let server = socket(AF_INET, libc::SOCK_STREAM, 6);
        assert_eq!(listen(server, 1), SOCKET_ERROR);
        assert_eq!(WSAGetLastError(), WSAEINVAL);
        assert_eq!(local_of(server), ([0; 4], 0));
        assert_eq!(bind(server, v4([127, 0, 0, 1], 0).as_ptr(), 16), 0);
        assert_eq!(listen(server, 1), 0);

        // A first datagram binds the sender, to loopback.
        let to = socket(AF_INET, libc::SOCK_DGRAM, 17);
        assert_eq!(bind(to, v4([127, 0, 0, 1], 0).as_ptr(), 16), 0);
        let (_, port) = local_of(to);
        let from = socket(AF_INET, libc::SOCK_DGRAM, 17);
        assert_eq!(sendto(from, b"x".as_ptr(), 1, 0, v4([127, 0, 0, 1], port).as_ptr(), 16), 1);
        let (ip, from_port) = local_of(from);
        assert_eq!(ip, [127, 0, 0, 1]);
        assert_ne!(from_port, 0);
        // A bound sender keeps its address.
        assert_eq!(sendto(from, b"y".as_ptr(), 1, 0, v4([127, 0, 0, 1], port).as_ptr(), 16), 1);
        assert_eq!(local_of(from), (ip, from_port));
        let mut got = [0u8; 4];
        assert_eq!(recv(to, got.as_mut_ptr(), 4, 0), 1);

        // The same for an IPv6 socket sending to 127.0.0.1.
        let six = socket(AF_INET6, libc::SOCK_DGRAM, 17);
        let mut mapped = [0u8; 28];
        mapped[..2].copy_from_slice(&(AF_INET6 as u16).to_le_bytes());
        mapped[2..4].copy_from_slice(&port.to_be_bytes());
        mapped[18..20].copy_from_slice(&[0xff, 0xff]);
        mapped[20..24].copy_from_slice(&[127, 0, 0, 1]);
        assert_eq!(sendto(six, b"z".as_ptr(), 1, 0, mapped.as_ptr(), 28), 1);
        let mut addr = [0u8; 28];
        let mut len = 28;
        assert_eq!(getsockname(six, addr.as_mut_ptr(), &mut len), 0);
        assert_eq!(&addr[8..24], &mapped[8..24]);

        for s in [six, from, to, server] {
            assert_eq!(closesocket(s), 0);
        }
    }

    #[test]
    fn names_resolve_offline() {
        start();
        let away = getaddrinfo(c"example.com".as_ptr(), c"443".as_ptr(), std::ptr::null(), &mut std::ptr::null_mut());
        assert_eq!(away, WSAHOST_NOT_FOUND);
        let hints = AddrInfo {
            flags: 0,
            family: AF_INET,
            kind: libc::SOCK_STREAM,
            protocol: 6,
            addr_len: 0,
            canonical: std::ptr::null_mut(),
            addr: std::ptr::null_mut(),
            next: std::ptr::null_mut(),
        };
        let mut res = std::ptr::null_mut();
        assert_eq!(getaddrinfo(c"localhost".as_ptr(), c"8080".as_ptr(), &hints, &mut res), 0);
        // SAFETY: the list getaddrinfo made.
        unsafe {
            let r = &*res;
            assert!(r.next.is_null());
            assert_eq!((r.family, r.addr_len), (AF_INET, 16));
            assert_eq!(std::slice::from_raw_parts(r.addr, 8), &v4([127, 0, 0, 1], 8080)[..8]);
        }
        freeaddrinfo(res);

        assert!(gethostbyname(c"example.com".as_ptr()).is_null());
        assert_eq!(WSAGetLastError(), WSAHOST_NOT_FOUND);
        let h = gethostbyname(c"127.0.0.1".as_ptr());
        // SAFETY: our per-thread hostent.
        unsafe {
            assert_eq!(((*h).kind, (*h).length), (2, 4));
            assert_eq!(std::slice::from_raw_parts(*(*h).addresses, 4), [127, 0, 0, 1]);
        }
        assert_eq!(inet_addr(c"127.0.0.1".as_ptr()), u32::from_ne_bytes([127, 0, 0, 1]));
        assert_eq!(inet_addr(c"nope".as_ptr()), u32::MAX);
        assert_eq!(htons(0x1234), 0x3412);
        let mut name = [0 as c_char; 256];
        assert_eq!(gethostname(name.as_mut_ptr(), 256), 0);
    }
}
