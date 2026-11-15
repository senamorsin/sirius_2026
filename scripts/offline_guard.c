/* Validation helper: deny non-loopback IP connections, including in child processes.
 * Build: gcc -shared -fPIC scripts/offline_guard.c -o output/offline_guard.so -ldl
 * Run Python with LD_PRELOAD=.../output/offline_guard.so. UNIX sockets remain local.
 */
#define _GNU_SOURCE
#include <arpa/inet.h>
#include <dlfcn.h>
#include <errno.h>
#include <netinet/in.h>
#include <sys/socket.h>

static int permitted(const struct sockaddr *address) {
    if (!address) return 1;
    if (address->sa_family == AF_INET) {
        const struct sockaddr_in *ip = (const struct sockaddr_in *) address;
        if ((ntohl(ip->sin_addr.s_addr) >> 24) != 127 || ntohs(ip->sin_port) == 53) return 0;
    }
    if (address->sa_family == AF_INET6) {
        const struct sockaddr_in6 *ip = (const struct sockaddr_in6 *) address;
        if (!IN6_IS_ADDR_LOOPBACK(&ip->sin6_addr) || ntohs(ip->sin6_port) == 53) return 0;
    }
    return 1;
}

int connect(int fd, const struct sockaddr *address, socklen_t size) {
    int (*original)(int, const struct sockaddr *, socklen_t) = dlsym(RTLD_NEXT, "connect");
    if (!permitted(address)) { errno = ENETUNREACH; return -1; }
    return original(fd, address, size);
}

ssize_t sendto(int fd, const void *buffer, size_t length, int flags,
               const struct sockaddr *address, socklen_t size) {
    ssize_t (*original)(int, const void *, size_t, int, const struct sockaddr *, socklen_t)
        = dlsym(RTLD_NEXT, "sendto");
    if (!permitted(address)) { errno = ENETUNREACH; return -1; }
    return original(fd, buffer, length, flags, address, size);
}
