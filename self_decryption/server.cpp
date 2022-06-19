#undef UNICODE

#define WIN32_LEAN_AND_MEAN

#include <windows.h>
#include <winsock2.h>
#include <ws2tcpip.h>
#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>

// Need to link with Ws2_32.lib
#pragma comment (lib, "Ws2_32.lib")
// #pragma comment (lib, "Mswsock.lib")

#define DEFAULT_PORT "27015"
#define NUM_FUNCS 2

enum seed {
    seed_a = 0x7facacac,
    seed_b = 0x00bdbdbd,

    error_keys = 0xFFFFFFFF,
};
const int32_t keys[NUM_FUNCS] = { seed_a, seed_b };
const int32_t error[NUM_FUNCS] = { error_keys, error_keys };
static constexpr int buflen = sizeof(int32_t) * NUM_FUNCS;

int32_t load_hashsum_from_file(char** argv) {
    FILE* f = nullptr;

    if (fopen_s(&f, argv[1], "r") || !f) {
        puts("Could not open file!");
        return -1;
    }
    int dummy, hashsum;
    fscanf_s(f, "%x %x", &dummy, &hashsum);
    fclose(f);
    printf("CRC32: 0x%x\n", hashsum);
    return hashsum;
}

int __cdecl main(int argc, char** argv)
{
    if (argc < 2) {
        puts("Please provide the file path!");
        return -1;
    }

    WSADATA wsaData;
    int iResult;
    alignas(sizeof(int32_t)) char recvbuf[sizeof(int32_t)];

    SOCKET ListenSocket = INVALID_SOCKET;
    SOCKET ClientSocket = INVALID_SOCKET;

    struct addrinfo* result = NULL;
    struct addrinfo hints;

    // Initialize Winsock
    iResult = WSAStartup(MAKEWORD(2, 2), &wsaData);
    if (iResult != 0) {
        printf("WSAStartup failed with error: %d\n", iResult);
        return 1;
    }

    ZeroMemory(&hints, sizeof(hints));
    hints.ai_family = AF_INET;
    hints.ai_socktype = SOCK_STREAM;
    hints.ai_protocol = IPPROTO_TCP;
    hints.ai_flags = AI_PASSIVE;

    // Resolve the server address and port
    iResult = getaddrinfo(NULL, DEFAULT_PORT, &hints, &result);
    if (iResult != 0) {
        printf("getaddrinfo failed with error: %d\n", iResult);
        WSACleanup();
        return 1;
    }

    // Create a SOCKET for connecting to server
    ListenSocket = socket(result->ai_family, result->ai_socktype, result->ai_protocol);
    if (ListenSocket == INVALID_SOCKET) {
        printf("socket failed with error: %ld\n", WSAGetLastError());
        freeaddrinfo(result);
        WSACleanup();
        return 1;
    }

    // Setup the TCP listening socket
    iResult = bind(ListenSocket, result->ai_addr, (int)result->ai_addrlen);
    if (iResult == SOCKET_ERROR) {
        printf("bind failed with error: %d\n", WSAGetLastError());
        freeaddrinfo(result);
        closesocket(ListenSocket);
        WSACleanup();
        return 1;
    }

    freeaddrinfo(result);

    iResult = listen(ListenSocket, SOMAXCONN);
    if (iResult == SOCKET_ERROR) {
        printf("listen failed with error: %d\n", WSAGetLastError());
        closesocket(ListenSocket);
        WSACleanup();
        return 1;
    }

    // Accept a client socket
    ClientSocket = accept(ListenSocket, NULL, NULL);
    if (ClientSocket == INVALID_SOCKET) {
        printf("accept failed with error: %d\n", WSAGetLastError());
        closesocket(ListenSocket);
        WSACleanup();
        return 1;
    }

    // No longer need server socket
    closesocket(ListenSocket);

    do {
        iResult = recv(ClientSocket, recvbuf, sizeof(recvbuf), 0);
        if (iResult > 0)
            printf("Bytes received: %d\n", iResult);
    } while (iResult > 0);

    const int32_t *seed = (*(int*)recvbuf == load_hashsum_from_file(argv)) ? keys : error;


    // Send the keys
    iResult = send(ClientSocket, (const char*)seed, buflen, 0);
    if (iResult == SOCKET_ERROR) {
        printf("send failed with error: %d\n", WSAGetLastError());
        closesocket(ClientSocket);
        WSACleanup();
        return 1;
    }

    printf("Bytes Sent: %ld\n", iResult);

    // shutdown the connection since no more data will be sent
    iResult = shutdown(ClientSocket, SD_SEND);
    if (iResult == SOCKET_ERROR) {
        printf("shutdown failed with error: %d\n", WSAGetLastError());
        closesocket(ClientSocket);
        WSACleanup();
        return 1;
    }

    // cleanup
    closesocket(ClientSocket);
    WSACleanup();

    return 0;
}