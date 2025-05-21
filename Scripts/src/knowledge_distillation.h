/**
 * knowledge_distill.h
 * Header file for knowledge distillation in darknet
 */

#ifndef KNOWLEDGE_DISTILL_H
#define KNOWLEDGE_DISTILL_H

#include "darknet.h"

// Knowledge distillation parameters structure
typedef struct {
    char *teacher_cfg;
    char *teacher_weights;
    char *student_cfg;
    char *student_weights;
    char *train_list;
    float temperature;
    int max_batches;
    int batch_size;
    int is_student_grayscale;
    // float distill_loss_weight;
    // float detection_threshold;
    float learning_rate;     // Learning rate for student network
    float momentum;          // Momentum for gradient updates
    float decay;             // Weight decay for regularization
} kd_params;

// Main function to run knowledge distillation
void run_knowledge_distill(int argc, char **argv);

// Helper functions
float *extract_detection_features(layer l, int *feature_size);
float compute_kd_loss(float *teacher_logits, float *student_logits, int size, float temperature);
layer find_detection_layer(network *net);
float process_distill_image(network *teacher, network *student, char *image_path, 
                          float temperature, int is_student_grayscale);
char **load_image_batch(char *train_list, int batch_size, int *num_loaded);
void train_knowledge_distillation(kd_params params);

#endif // KNOWLEDGE_DISTILL_H