/**
 * knowledge_distill.c
 * Knowledge distillation implementation as an extension to darknet
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>
#include "knowledge_distillation.h"
#include "blas.h"
#include "darknet.h"
#include "network.h"
#include "parser.h"

// Reset gradients before accumulating for a new batch
static void reset_network_gradients(network net) {
    int i;
    for (i = 0; i < net.n; ++i) {
        layer l = net.layers[i];
        if (l.delta) {
            // Reset deltas to zero
            memset(l.delta, 0, l.outputs * sizeof(float));
        }
    }
}

// Scale gradients by a factor (e.g., for batch normalization)
static void scale_network_gradients(network net, float scale_factor) {
    int i;
    for (i = 0; i < net.n; ++i) {
        layer l = net.layers[i];
        if (l.delta) {
            // Scale deltas by the factor
            int j;
            for (j = 0; j < l.outputs; ++j) {
                l.delta[j] *= scale_factor;
            }
        }
    }
}

// Function for backward propagation from a specific layer
static void backward_network_from(network net, int start) {
    int i;
    network_state state = {0};
    state.net = net;
    state.train = 1;
    
    for (i = start; i >= 0; --i) {
        layer l = net.layers[i];
        if (l.backward) {
            state.index = i;
            l.backward(l, state);
        }
    }
}

// Function to compute gradients for KL divergence
static void compute_kd_gradient(float *teacher_logits, float *student_logits, float *gradient, int size, float temperature) {
    if (!teacher_logits || !student_logits || !gradient || size <= 0) {
        return;
    }
    
    // Create copies of the logits to apply temperature
    float *t_probs = calloc(size, sizeof(float));
    float *s_probs = calloc(size, sizeof(float));
    
    if (!t_probs || !s_probs) {
        if (t_probs) free(t_probs);
        if (s_probs) free(s_probs);
        return;
    }
    
    // Apply softmax with temperature to both (using your softmax function)
    softmax(teacher_logits, size, temperature, t_probs, 1);
    softmax(student_logits, size, temperature, s_probs, 1);
    
    // Compute gradient of KL divergence with respect to student logits
    // For KL(t||s), the gradient is: s_probs - t_probs
    // Note: This is multiplied by 1/temperature because we're using softmax with temperature
    float epsilon = 1e-7f;
    for (int i = 0; i < size; i++) {
        // The gradient of KL divergence with respect to student logits
        // is proportional to the difference between the probabilities
        gradient[i] = (s_probs[i] - t_probs[i]) / temperature;
    }
    
    // Clean up
    free(t_probs);
    free(s_probs);
}

// Extract features from detection layer for knowledge distillation
float *extract_detection_features(layer l, int *feature_size) {
    if (!l.output || l.outputs <= 0) {
        *feature_size = 0;
        return NULL;
    }
    
    *feature_size = l.outputs;
    float *features = calloc(*feature_size, sizeof(float));
    if (!features) {
        fprintf(stderr, "Memory allocation failed for detection features\n");
        *feature_size = 0;
        return NULL;
    }
    
    // Copy layer outputs directly
    memcpy(features, l.output, l.outputs * sizeof(float));
    return features;
}

// Apply temperature to logits
void apply_temperature(float *logits, int size, float temperature) {
    for (int i = 0; i < size; i++) {
        logits[i] /= temperature;
    }
}

// Compute KL divergence loss for knowledge distillation
float compute_kd_loss(float *teacher_logits, float *student_logits, int size, float temperature) {
    if (!teacher_logits || !student_logits || size <= 0) {
        return 0.0f;
    }

    // Create copies of the logits to apply temperature
    float *t_logits = calloc(size, sizeof(float));
    float *s_logits = calloc(size, sizeof(float));
    float *t_probs = calloc(size, sizeof(float));
    float *s_probs = calloc(size, sizeof(float));
    
    if (!t_logits || !s_logits || !t_probs || !s_probs) {
        if (t_logits) free(t_logits);
        if (s_logits) free(s_logits);
        if (t_probs) free(t_probs);
        if (s_probs) free(s_probs);
        return 0.0f;
    }
    
    memcpy(t_logits, teacher_logits, size * sizeof(float));
    memcpy(s_logits, student_logits, size * sizeof(float));

    // Apply softmax with temperature to both
    softmax(t_logits, size, temperature, t_probs, 1);
    softmax(s_logits, size, temperature, s_probs, 1);

    // Compute KL divergence: sum(t_probs * log(t_probs / s_probs))
    float loss = 0.0f;
    float epsilon = 1e-7f;
    for (int i = 0; i < size; i++) {
        if (t_probs[i] > epsilon) {
            loss += t_probs[i] * log((t_probs[i] + epsilon) / (s_probs[i] + epsilon));
        }
    }

    // Clean up
    free(t_logits);
    free(s_logits);
    free(t_probs);
    free(s_probs);
    return loss;
}

// Find the detection layer in a network
layer find_detection_layer(network *net) {
    int i;
    for (i = net->n - 1; i >= 0; i--) {
        if (net->layers[i].type == DETECTION) {
            return net->layers[i];
        } else if (net->layers[i].type == YOLO) {
            return net->layers[i];
        }
    }
    // Return empty layer if not found
    layer l = {0};
    fprintf(stderr, "Warning: No detection/YOLO layer found in network\n");
    return l;
}

// Process one image through both teacher and student networks
float process_distill_image(network *teacher, network *student, char *image_path, 
                          float temperature, int is_student_grayscale) {
    image im;
    
    // Load image for teacher (always color)
    im = load_image_color(image_path, 0, 0);
    if (!im.data) {
        fprintf(stderr, "Error: Could not read image %s\n", image_path);
        return 0.0f;
    }
    
    // Resize image to network dimensions
    image sized = letterbox_image(im, teacher->w, teacher->h);
    
    // Run teacher network
    network_predict(*teacher, sized.data);
    free_image(sized);
    
    // Get teacher detection layer
    layer teacher_layer = find_detection_layer(teacher);
    if (teacher_layer.outputs <= 0) {
        free_image(im);
        return 0.0f;
    }
    
    // Extract teacher features
    int teacher_feature_size = 0;
    float *teacher_features = extract_detection_features(teacher_layer, &teacher_feature_size);
    
    // Load image for student (grayscale if specified)
    image im_student;
    if (is_student_grayscale) {
        free_image(im);  // Free the color image
        im_student = load_image(image_path, 0, 0, 1);
        if (!im_student.data) {
            fprintf(stderr, "Error: Could not read grayscale image %s\n", image_path);
            if (teacher_features) free(teacher_features);
            return 0.0f;
        }
        // Convert grayscale to 3 channels for darknet
        image rgb = make_image(im_student.w, im_student.h, 3);
        for (int i = 0; i < im_student.w * im_student.h; ++i) {
            rgb.data[i] = im_student.data[i];
            rgb.data[i + im_student.w * im_student.h] = im_student.data[i];
            rgb.data[i + 2 * im_student.w * im_student.h] = im_student.data[i];
        }
        free_image(im_student);
        im_student = rgb;
    } else {
        im_student = copy_image(im);
    }
    
    // Process student network with forward pass
    sized = letterbox_image(im_student, student->w, student->h);
    
    // Forward pass (we need this to happen in train mode)
    network_state state = {0};
    state.net = *student;
    state.index = 0;
    state.train = 1;
    state.input = sized.data;
    
    forward_network(*student, state);
    
    // Get student detection layer
    layer student_layer = find_detection_layer(student);
    if (student_layer.outputs <= 0) {
        free_image(sized);
        free_image(im_student);
        if (teacher_features) free(teacher_features);
        return 0.0f;
    }
    
    // Extract student features
    int student_feature_size = 0;
    float *student_features = extract_detection_features(student_layer, &student_feature_size);
    
    // Compute KL divergence loss
    float loss = 0.0f;
    if (teacher_features && student_features && 
        teacher_feature_size > 0 && student_feature_size > 0) {
        
        // Make sure we have the same feature size
        int min_size = (teacher_feature_size < student_feature_size) ? 
                      teacher_feature_size : student_feature_size;
        
        // Compute loss using your KL divergence function
        loss = compute_kd_loss(teacher_features, student_features, min_size, temperature);
        
        if (loss > 0) {
            // Backpropagate the loss
            // Create deltas (gradients) for the student network
            float *deltas = calloc(min_size, sizeof(float));
            if (!deltas) {
                fprintf(stderr, "Error: Failed to allocate memory for gradients\n");
                return loss;
            }
            
            // Calculate gradients based on your KL divergence loss
            compute_kd_gradient(teacher_features, student_features, deltas, min_size, temperature);
            
            // Apply the gradients to the student layer
            // First, find which layer in the student network corresponds to our extracted features
            int layer_index = student_layer.index;
            
            // Set the deltas for the student layer
            if (layer_index >= 0 && layer_index < student->n && 
                student->layers[layer_index].delta != NULL) {
                
                // Copy our calculated gradients to the layer's delta
                // Note: This might need adjustment based on how features are extracted
                // and how deltas are structured in your network
                memcpy(student->layers[layer_index].delta, deltas, min_size * sizeof(float));
                
                // Backward pass through the student network starting from this layer
                backward_network_from(*student, layer_index);
            } else {
                fprintf(stderr, "Warning: Could not find appropriate layer for backpropagation\n");
            }
            
            // Free the delta buffer
            free(deltas);
        }
    }
    
    // Clean up
    free_image(sized);
    free_image(im_student);
    if (teacher_features) free(teacher_features);
    if (student_features) free(student_features);
    
    return loss;
}

// Load a batch of image paths
char **load_image_batch(char *train_list, int batch_size, int *num_loaded) {
    FILE *file = fopen(train_list, "r");
    if (!file) {
        fprintf(stderr, "Error opening file: %s\n", train_list);
        *num_loaded = 0;
        return NULL;
    }

    // Count lines in file
    int count = 0;
    char c;
    while ((c = fgetc(file)) != EOF) {
        if (c == '\n') count++;
    }
    // Add one more if file doesn't end with newline
    if (count == 0 || c != '\n') count++;

    if (count == 0) {
        fclose(file);
        *num_loaded = 0;
        return NULL;
    }

    // Reset file pointer
    rewind(file);

    // Read all paths
    char **all_paths = calloc(count, sizeof(char*));
    if (!all_paths) {
        fprintf(stderr, "Memory allocation failed\n");
        fclose(file);
        *num_loaded = 0;
        return NULL;
    }

    char line[256];
    int i = 0;
    while (fgets(line, sizeof(line), file) && i < count) {
        // Remove newline
        size_t len = strlen(line);
        if (len > 0 && line[len-1] == '\n') {
            line[len-1] = '\0';
        }
        all_paths[i] = strdup(line);
        if (!all_paths[i]) {
            // Memory allocation failed, clean up
            for (int j = 0; j < i; j++) {
                free(all_paths[j]);
            }
            free(all_paths);
            fclose(file);
            *num_loaded = 0;
            return NULL;
        }
        i++;
    }
    fclose(file);

    // The actual number of paths read might be less than count
    int actual_count = i;

    // Randomly select batch_size paths
    int size = (batch_size < actual_count) ? batch_size : actual_count;
    char **batch_paths = calloc(size, sizeof(char*));
    if (!batch_paths) {
        // Memory allocation failed, clean up
        for (i = 0; i < actual_count; i++) {
            free(all_paths[i]);
        }
        free(all_paths);
        *num_loaded = 0;
        return NULL;
    }

    // Seed random number generator
    srand((unsigned int)time(NULL));

    // Fisher-Yates shuffle algorithm to select random paths
    for (i = 0; i < size; i++) {
        int j = i + rand() % (actual_count - i);
        
        // Swap all_paths[i] and all_paths[j]
        char *temp = all_paths[i];
        all_paths[i] = all_paths[j];
        all_paths[j] = temp;
        
        // Copy the first 'size' elements to batch_paths
        batch_paths[i] = strdup(all_paths[i]);
        if (!batch_paths[i]) {
            // Memory allocation failed, clean up
            for (int k = 0; k < i; k++) {
                free(batch_paths[k]);
            }
            free(batch_paths);
            for (int k = 0; k < actual_count; k++) {
                free(all_paths[k]);
            }
            free(all_paths);
            *num_loaded = 0;
            return NULL;
        }
    }

    // Free all paths in all_paths array
    for (i = 0; i < actual_count; i++) {
        free(all_paths[i]);
    }
    free(all_paths);

    *num_loaded = size;
    return batch_paths;
}

// Main knowledge distillation training function
void train_knowledge_distillation(kd_params params) {
    printf("Loading teacher model: %s, %s\n", params.teacher_cfg, params.teacher_weights);
    network *teacher = load_network(params.teacher_cfg, params.teacher_weights, 0);
    printf("Loading student model: %s, %s\n", params.student_cfg, params.student_weights);
    network *student = load_network(params.student_cfg, params.student_weights, 0);
    
    if (!teacher || !student) {
        fprintf(stderr, "Failed to load one or both networks\n");
        if (teacher) free_network(*teacher);
        if (student) free_network(*student);
        return;
    }
    
    printf("Training with data from %s\n", params.train_list);
    printf("Temperature: %f, Max batches: %d, Batch size: %d\n", 
          params.temperature, params.max_batches, params.batch_size);
    printf("Learning rate: %f, Momentum: %f, Decay: %f\n",
          params.learning_rate, params.momentum, params.decay);
    
    // Set both networks to train mode (student needs to be in train mode)
    teacher->train = 0;  // Teacher is frozen, only used for forward passes
    student->train = 1;  // Student is in training mode
    
    // Configure student network optimizer parameters
    student->learning_rate = params.learning_rate;
    student->momentum = params.momentum;
    student->decay = params.decay;
    
    // Create output directory if it doesn't exist
    #ifdef _WIN32
    system("if not exist out mkdir out");
    #else
    system("mkdir -p out");
    #endif
    
    // Training loop
    int i;
    for (i = 0; i < params.max_batches; i++) {
        clock_t time = clock();
        
        // Load a batch of images
        int batch_size = 0;
        char **batch_paths = load_image_batch("/home/ngerr/Workspace/Thesis/YOLOFastest/Yolo-Fastest/helmet_datasetv2_1/train.txt", params.batch_size, &batch_size);
        if (!batch_paths || batch_size == 0) {
            fprintf(stderr, "Failed to load image batch\n");
            continue;
        }
        
        float batch_loss = 0.0f;
        int valid_samples = 0;
        
        // Reset accumulated gradients for the batch
        reset_network_gradients(*student);
        
        // Process each image in the batch
        int j;
        for (j = 0; j < batch_size; j++) {
            // Forward and backward pass, computing loss
            float loss = process_distill_image(teacher, student, batch_paths[j], 
                                          params.temperature, params.is_student_grayscale);
            if (loss > 0) {
                batch_loss += loss;
                valid_samples++;
            }
        }
        
        if (valid_samples > 0) {
            // Scale the gradients by batch size for proper averaging
            scale_network_gradients(*student, 1.0f / valid_samples);
            
            // Update weights with the calculated gradients
            update_network(*student);
        }
        
        // Calculate average loss
        float avg_loss = (valid_samples > 0) ? (batch_loss / valid_samples) : 0.0f;
        
        // Calculate time taken
        clock_t elapsed = clock() - time;
        float sec = (float)elapsed / CLOCKS_PER_SEC;
        
        // Print progress
        printf("Batch %d/%d, Loss: %f, Valid Samples: %d/%d, Time: %.3f sec\n", 
              i+1, params.max_batches, avg_loss, valid_samples, batch_size, sec);
        
        // Reduce learning rate periodically (optional)
        if ((i + 1) % 100 == 0) {
            student->learning_rate *= 0.9;
            printf("Learning rate adjusted to %f\n", student->learning_rate);
        }
        
        // Save checkpoint periodically
        if ((i + 1) % 5 == 0 || (i + 1) == params.max_batches) {
            char buff[256];
            sprintf(buff, "out/student_kd_checkpoint_%d.weights", i+1);
            save_weights(*student, buff);
        }
        
        // Free batch paths
        for (j = 0; j < batch_size; j++) {
            free(batch_paths[j]);
        }
        free(batch_paths);
    }
    
    // Save final model
    printf("Saving final distilled model...\n");
    save_weights(*student, "student_distilled_final.weights");
    printf("Knowledge distillation complete!\n");
    
    // Clean up
    free_network(*teacher);
    free_network(*student);
}

// Parse command line arguments and run knowledge distillation
void run_knowledge_distill(int argc, char **argv) {
    // Set default parameters
    kd_params params = {
        .teacher_cfg = "mycfg/yolov4-tiny_helmet.cfg",
        .teacher_weights = "mybackup/yolov4-tiny_helmet_final.weights",
        .student_cfg = "mycfg/yolofv1_helmetv2_reduce_filter_gray.cfg",
        .student_weights = "mybackup/yolofv1_helmetv2_reduce_filter_gray_best.weights",
        .train_list = NULL,
        .temperature = 2.0,
        .max_batches = 10000,
        .batch_size = 16,
        .is_student_grayscale = 1,
        // .distill_loss_weight = 0.5,
        // .detection_threshold = 0.5
        .learning_rate = 0.0001,
        .momentum = 0,
        .decay = 0,
    };
    
    // Parse arguments
    int i;
    for (i = 1; i < argc; i++) {
        if (i + 1 < argc) {
            if (strcmp(argv[i], "-teacher_cfg") == 0) params.teacher_cfg = argv[++i];
            else if (strcmp(argv[i], "-teacher_weights") == 0) params.teacher_weights = argv[++i];
            else if (strcmp(argv[i], "-student_cfg") == 0) params.student_cfg = argv[++i];
            else if (strcmp(argv[i], "-student_weights") == 0) params.student_weights = argv[++i];
            else if (strcmp(argv[i], "-train_list") == 0) params.train_list = argv[++i];
            else if (strcmp(argv[i], "-temperature") == 0) params.temperature = atof(argv[++i]);
            else if (strcmp(argv[i], "-max_batches") == 0) params.max_batches = atoi(argv[++i]);
            else if (strcmp(argv[i], "-batch_size") == 0) params.batch_size = atoi(argv[++i]);
            else if (strcmp(argv[i], "-gray") == 0) params.is_student_grayscale = atoi(argv[++i]);
            // else if (strcmp(argv[i], "-distill_weight") == 0) params.distill_loss_weight = atof(argv[++i]);
            // else if (strcmp(argv[i], "-threshold") == 0) params.detection_threshold = atof(argv[++i]);
        }
    }
    
    // Extract train list path from data config if not provided
    if (!params.train_list) {
        char *data_cfg = "helmet_datasetv2_1/helmetv2.data";
        FILE *file = fopen(data_cfg, "r");
        if (file) {
            char line[256];
            while (fgets(line, sizeof(line), file)) {
                if (strncmp(line, "train", 5) == 0) {
                    char *equals = strchr(line, '=');
                    if (equals) {
                        // Skip whitespace after equals sign
                        char *path_start = equals + 1;
                        while (*path_start == ' ' || *path_start == '\t') {
                            path_start++;
                        }
                        
                        // Remove trailing whitespace and newline
                        size_t len = strlen(path_start);
                        while (len > 0 && (path_start[len-1] == '\n' || path_start[len-1] == ' ' || 
                                          path_start[len-1] == '\t')) {
                            len--;
                        }
                        
                        params.train_list = calloc(len + 1, sizeof(char));
                        if (params.train_list) {
                            strncpy(params.train_list, path_start, len);
                            params.train_list[len] = '\0';
                        }
                        break;
                    }
                }
            }
            fclose(file);
        }
    }
    
    // Check that we have the training list
    if (!params.train_list) {
        fprintf(stderr, "Error: No training list provided or found in data config\n");
        return;
    }
    
    // Run training
    train_knowledge_distillation(params);
    
    // Free allocated memory if needed
    if (params.train_list && (params.train_list != argv[i-1])) {
        free(params.train_list);
    }
}