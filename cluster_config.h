#pragma once
#include <stdint.h>

// --- LLM Model Configuration ---
#define HIDDEN_DIM 288       // E.g., for a 15M TinyStories model
#define VOCAB_SIZE 32000
#define MAX_CONTEXT_LENGTH 64
#define EOS_TOKEN_ID 2       // End of Sequence ID

// --- Network Specifications ---
#define NETWORK_TIMEOUT_MS 150
#define MAX_ESP_NOW_PAYLOAD 250
#define FLOATS_PER_PACKET 60 // 240 bytes

// --- Message Types ---
#define MSG_TYPE_TENSOR 0
#define MSG_TYPE_FEEDBACK 1
#define MSG_TYPE_WIPE 2
#define MSG_TYPE_FAULT 3

// --- Packet Structures ---
typedef struct __attribute__((packed)) {
    uint8_t message_type;
    uint8_t sender_id;
    uint32_t token_id;
    uint16_t packet_index;
    uint16_t total_packets;
} TensorHeader;

typedef struct __attribute__((packed)) {
    uint8_t message_type;
    uint32_t token_id;
} TokenFeedbackPacket;

typedef struct __attribute__((packed)) {
    uint8_t message_type;
    uint8_t failing_node_id;
} FaultPacket;

// --- Physical Routing Table ---
const uint8_t broadcast_mac[6] = {0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF};

// Replace these with the actual MAC addresses of your 8 ESP32-S3 boards
const uint8_t mac_routing_table[8][6] = {
    {0xAA, 0xBB, 0xCC, 0xDD, 0xEE, 0x01}, // Node 1 (Head)
    {0xAA, 0xBB, 0xCC, 0xDD, 0xEE, 0x02}, // Node 2
    {0xAA, 0xBB, 0xCC, 0xDD, 0xEE, 0x03}, // Node 3
    {0xAA, 0xBB, 0xCC, 0xDD, 0xEE, 0x04}, // Node 4
    {0xAA, 0xBB, 0xCC, 0xDD, 0xEE, 0x05}, // Node 5
    {0xAA, 0xBB, 0xCC, 0xDD, 0xEE, 0x06}, // Node 6
    {0xAA, 0xBB, 0xCC, 0xDD, 0xEE, 0x07}, // Node 7
    {0xAA, 0xBB, 0xCC, 0xDD, 0xEE, 0x08}  // Node 8 (Tail)
};
