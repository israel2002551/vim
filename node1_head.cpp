#include <Arduino.h>
#include <WiFi.h>
#include <esp_now.h>
#include "esp_partition.h"
#include "cluster_config.h"

#define NODE_ID 1
#define NEXT_NODE_ID 2

const float* mapped_embedding_table = NULL;

volatile bool generate_next_word = false;
volatile uint32_t next_token_id = 0;

void transmit_to_next_node(const float* tensor, uint32_t token_id) {
    uint8_t tx_buffer[MAX_ESP_NOW_PAYLOAD];
    int total_packets = (HIDDEN_DIM + FLOATS_PER_PACKET - 1) / FLOATS_PER_PACKET;
    
    for (int i = 0; i < total_packets; i++) {
        TensorHeader header = {MSG_TYPE_TENSOR, NODE_ID, token_id, (uint16_t)i, (uint16_t)total_packets};
        memcpy(tx_buffer, &header, sizeof(TensorHeader));
        
        int floats_remaining = HIDDEN_DIM - (i * FLOATS_PER_PACKET);
        int floats_to_send = (floats_remaining > FLOATS_PER_PACKET) ? FLOATS_PER_PACKET : floats_remaining;
        
        memcpy(tx_buffer + sizeof(TensorHeader), &tensor[i * FLOATS_PER_PACKET], floats_to_send * sizeof(float));
        esp_now_send(mac_routing_table[NEXT_NODE_ID - 1], tx_buffer, sizeof(TensorHeader) + (floats_to_send * sizeof(float)));
        delay(2); 
    }
}

void OnDataRecv(const uint8_t *mac, const uint8_t *incomingData, int len) {
    if (len == sizeof(TokenFeedbackPacket) && incomingData[0] == MSG_TYPE_FEEDBACK) {
        TokenFeedbackPacket feedback;
        memcpy(&feedback, incomingData, sizeof(TokenFeedbackPacket));
        next_token_id = feedback.token_id;
        generate_next_word = true;
    }
    
    else if (len == sizeof(FaultPacket) && incomingData[0] == MSG_TYPE_FAULT) {
        FaultPacket fault;
        memcpy(&fault, incomingData, sizeof(FaultPacket));
        Serial.printf("\n[FAULT] Node %d timed out! Generation Aborted.\n", fault.failing_node_id);
        generate_next_word = false; // Kill engine
    }
}

void setup() {
    Serial.begin(115200);
    WiFi.mode(WIFI_STA);
    esp_now_init();
    esp_now_register_recv_cb(OnDataRecv);
    
    // Register peers (Node 2 and Broadcast)
    esp_now_peer_info_t peerInfo = {};
    peerInfo.channel = 0; peerInfo.encrypt = false;
    
    memcpy(peerInfo.peer_addr, mac_routing_table[NEXT_NODE_ID - 1], 6);
    esp_now_add_peer(&peerInfo);
    
    memcpy(peerInfo.peer_addr, broadcast_mac, 6);
    esp_now_add_peer(&peerInfo);

    esp_partition_mmap_handle_t mmap_handle;
    const esp_partition_t* partition = esp_partition_find_first(ESP_PARTITION_TYPE_DATA, ESP_PARTITION_SUBTYPE_DATA_SPIFFS, "embeddings");
    if (partition) {
        const void* map_ptr;
        esp_partition_mmap(partition, 0, partition->size, ESP_PARTITION_MMAP_DATA, &map_ptr, &mmap_handle);
        mapped_embedding_table = (const float*) map_ptr;
        Serial.println("Node 1 Ready. Type a prompt:");
    }
}

// Mock Tokenizer for structure
int tokenize_string(String input, uint32_t* output_tokens) {
    output_tokens[0] = 10943; // E.g., "Hello"
    return 1; 
}

void loop() {
    // 1. Initial Prompt Execution
    if (Serial.available()) {
        String prompt = Serial.readStringUntil('\n');
        
        // Broadcast Cache Wipe
        uint8_t wipe_cmd = MSG_TYPE_WIPE;
        esp_now_send(broadcast_mac, &wipe_cmd, 1);
        delay(15); 
        
        uint32_t token_ids[16]; 
        int token_count = tokenize_string(prompt, token_ids);
        
        for (int i = 0; i < token_count; i++) {
            const float* initial_tensor = &mapped_embedding_table[token_ids[i] * HIDDEN_DIM];
            transmit_to_next_node(initial_tensor, token_ids[i]);
            delay(150); // Provide network breathing room during prompt loading
        }
    }
    
    // 2. Autoregressive Loop Execution
    if (generate_next_word) {
        generate_next_word = false;
        const float* next_tensor = &mapped_embedding_table[next_token_id * HIDDEN_DIM];
        transmit_to_next_node(next_tensor, next_token_id);
    }
}
