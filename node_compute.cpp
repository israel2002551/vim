#include <Arduino.h>
#include <WiFi.h>
#include <esp_now.h>
#include "esp_partition.h"
#include "dl_math.hpp"
#include "dl_tensor.hpp"
#include "cluster_config.h"

// --- CHANGE THIS FOR EACH BOARD (2 through 7) ---
#define NODE_ID 2
#define NEXT_NODE_ID (NODE_ID + 1)
// ------------------------------------------------

float current_activation[HIDDEN_DIM];
float output_activation[HIDDEN_DIM];

// Network & Timeout State
volatile bool tensor_ready = false;
volatile bool receiving_tensor = false;
volatile uint16_t packets_received = 0;
volatile uint32_t last_packet_time = 0;

// Context State
uint16_t current_token_index = 0;
float k_cache[MAX_CONTEXT_LENGTH * HIDDEN_DIM];
float v_cache[MAX_CONTEXT_LENGTH * HIDDEN_DIM];

const float* mapped_weights = NULL;
uint32_t current_token_id = 0;

void transmit_to_next_node(float* tensor, uint32_t token_id) {
    uint8_t tx_buffer[MAX_ESP_NOW_PAYLOAD];
    int total_packets = (HIDDEN_DIM + FLOATS_PER_PACKET - 1) / FLOATS_PER_PACKET;
    
    for (int i = 0; i < total_packets; i++) {
        TensorHeader header = {MSG_TYPE_TENSOR, NODE_ID, token_id, (uint16_t)i, (uint16_t)total_packets};
        memcpy(tx_buffer, &header, sizeof(TensorHeader));
        
        int floats_to_send = min(FLOATS_PER_PACKET, HIDDEN_DIM - (i * FLOATS_PER_PACKET));
        memcpy(tx_buffer + sizeof(TensorHeader), &tensor[i * FLOATS_PER_PACKET], floats_to_send * sizeof(float));
        
        esp_now_send(mac_routing_table[NEXT_NODE_ID - 1], tx_buffer, sizeof(TensorHeader) + (floats_to_send * sizeof(float)));
        delay(2);
    }
}

void OnDataRecv(const uint8_t *mac, const uint8_t *incomingData, int len) {
    
    // 1. Wipe Command
    if (len == 1 && incomingData[0] == MSG_TYPE_WIPE) {
        current_token_index = 0;
        tensor_ready = false; receiving_tensor = false; packets_received = 0;
        return;
    }
    
    // 2. Fault Command
    if (len == sizeof(FaultPacket) && incomingData[0] == MSG_TYPE_FAULT) {
        FaultPacket fault;
        memcpy(&fault, incomingData, sizeof(FaultPacket));
        if (fault.failing_node_id != NODE_ID) {
            tensor_ready = false; receiving_tensor = false; packets_received = 0;
        }
        return;
    }
    
    // 3. Tensor Assembly
    if (incomingData[0] == MSG_TYPE_TENSOR) {
        TensorHeader header;
        memcpy(&header, incomingData, sizeof(TensorHeader));
        
        last_packet_time = millis();
        receiving_tensor = true;
        current_token_id = header.token_id;
        
        int bytes_received = len - sizeof(TensorHeader);
        memcpy(&current_activation[header.packet_index * FLOATS_PER_PACKET], 
               incomingData + sizeof(TensorHeader), bytes_received);
        
        packets_received++;
        if (packets_received >= header.total_packets) {
            tensor_ready = true;
            receiving_tensor = false;
        }
    }
}

void setup() {
    Serial.begin(115200);
    WiFi.mode(WIFI_STA);
    esp_now_init();
    esp_now_register_recv_cb(OnDataRecv);
    
    esp_now_peer_info_t peer = {};
    peer.channel = 0; peer.encrypt = false;
    
    memcpy(peer.peer_addr, mac_routing_table[NEXT_NODE_ID - 1], 6);
    esp_now_add_peer(&peer);
    
    memcpy(peer.peer_addr, broadcast_mac, 6);
    esp_now_add_peer(&peer);

    esp_partition_mmap_handle_t mmap_handle;
    const esp_partition_t* partition = esp_partition_find_first(ESP_PARTITION_TYPE_DATA, ESP_PARTITION_SUBTYPE_DATA_SPIFFS, "model");
    if (partition) {
        const void* map_ptr;
        esp_partition_mmap(partition, 0, partition->size, ESP_PARTITION_MMAP_DATA, &map_ptr, &mmap_handle);
        mapped_weights = (const float*) map_ptr;
        Serial.printf("Node %d Weights Mapped.\n", NODE_ID);
    }
}

void loop() {
    
    // Watchdog Trigger
    if (receiving_tensor && (millis() - last_packet_time > NETWORK_TIMEOUT_MS)) {
        Serial.printf("Timeout on Node %d!\n", NODE_ID);
        
        FaultPacket fault = {MSG_TYPE_FAULT, NODE_ID};
        esp_now_send(broadcast_mac, (uint8_t*)&fault, sizeof(FaultPacket));
        
        receiving_tensor = false; tensor_ready = false; packets_received = 0;
    }

    if (tensor_ready) {
        
        // Wrap for ESP-DL Math Engine
        dl::Tensor<float> input_tensor;
        input_tensor.set_element((float *)current_activation).set_shape({1, 1, 1, HIDDEN_DIM}).set_auto_free(false);

        dl::Tensor<float> weights_tensor;
        weights_tensor.set_element((float *)mapped_weights).set_shape({1, 1, HIDDEN_DIM, HIDDEN_DIM}).set_auto_free(false);

        dl::Tensor<float> output_tensor;
        output_tensor.set_element((float *)output_activation).set_shape({1, 1, 1, HIDDEN_DIM}).set_auto_free(false);

        dl::math::matmul(&output_tensor, &input_tensor, &weights_tensor);
        
        // (KV Cache & Attention execution omitted for brevity)
        
        transmit_to_next_node(output_activation, current_token_id);
        
        if (current_token_index < MAX_CONTEXT_LENGTH) current_token_index++;
        tensor_ready = false; packets_received = 0;
    }
}
