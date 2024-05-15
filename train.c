// --- header files ---
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

// --- constants ---
#define NUMBER_OF_EXAMPLES 60000
#define IMAGE_SIZE (28 * 28)
#define NUMBER_OF_POSSIBILITIES 10

#define LEARNING_RATE 0.01
#define BATCH_SIZE 60

#define HIDDEN_LAYER_1_SIZE 512
#define HIDDEN_LAYER_2_SIZE 256

// --- structs ---
struct parameters {
    float weights_1[HIDDEN_LAYER_1_SIZE * IMAGE_SIZE]; // matrix
    float biases_1[HIDDEN_LAYER_1_SIZE]; // vector

    float weights_2[HIDDEN_LAYER_2_SIZE * HIDDEN_LAYER_1_SIZE]; // matrix
    float biases_2[HIDDEN_LAYER_2_SIZE]; // vector

    float weights_output[NUMBER_OF_POSSIBILITIES * HIDDEN_LAYER_2_SIZE]; // matrix
    float biases_output[NUMBER_OF_POSSIBILITIES]; // vector
};

struct dataset {
    float images[NUMBER_OF_EXAMPLES * IMAGE_SIZE];
    int labels[NUMBER_OF_EXAMPLES * NUMBER_OF_POSSIBILITIES];
};

struct gradients {
    float delta_weights_1[HIDDEN_LAYER_1_SIZE * IMAGE_SIZE]; // matrix
    float delta_biases_1[HIDDEN_LAYER_1_SIZE]; // vector

    float delta_weights_2[HIDDEN_LAYER_2_SIZE * HIDDEN_LAYER_1_SIZE]; // matrix
    float delta_biases_2[HIDDEN_LAYER_2_SIZE]; // vector

    float delta_weights_output[NUMBER_OF_POSSIBILITIES * HIDDEN_LAYER_2_SIZE]; // matrix
    float delta_biases_output[NUMBER_OF_POSSIBILITIES]; // vector
};

// --- preprocess the images and labels from the training data  ---
void preprocess_images(char path_to_images[], struct dataset *s) {
    FILE *file;
    int magic_number, size_of_file;
    unsigned char *buffer_file;

    // open training images as binary and store them in file
    file = fopen(path_to_images, "rb");
    if (file == NULL) {
        printf("Error: Unable to open file %s\n", path_to_images);
        exit(EXIT_FAILURE);
    }

    // read magic number from the header of the file, and convert to little-endian
    if (fread(&magic_number, sizeof(magic_number), 1, file) < 1) {
        printf("Error: Failed to read image data from file %s\n", path_to_images);
        exit(EXIT_FAILURE);
    }

    if (__builtin_bswap32(magic_number) != 2051) {
        printf("Error: Unrecognisable file type %s\n", path_to_images);
        exit(EXIT_FAILURE);
    }

    // find the size of the file, and go to the end of the header
    fseek(file, 0, SEEK_END);
    size_of_file = ftell(file) - 16;
    fseek(file, 16, SEEK_SET);

    // read the file into a buffer
    buffer_file = malloc(size_of_file * sizeof(unsigned char));
    if (buffer_file == NULL) {
        printf("Error: Memory allocation failed during preprocessing\n");
        exit(EXIT_FAILURE);
    }

    fread(buffer_file, size_of_file, 1, file);

    // normalize pixel values of the training images
    for (int i = 0; i < (NUMBER_OF_EXAMPLES * IMAGE_SIZE); i++) {
        s->images[i] = buffer_file[i] / 255.0;
    }

    free(buffer_file);
    fclose(file);
}

void preprocess_labels(char path_to_labels[], struct dataset *s) {
    FILE *file;
    int magic_number, size_of_file;
    unsigned char *buffer_file;

    // open training labels as binary and store them in file
    file = fopen(path_to_labels, "rb");
    if (file == NULL) {
        printf("Error: Unable to open file %s\n", path_to_labels);
        exit(EXIT_FAILURE);
    }

    // read magic number from header, and convert to little-endian
    if (fread(&magic_number, sizeof(magic_number), 1, file) < 1) {
        printf("Error: Failed to read label data from file %s\n", path_to_labels);
        exit(EXIT_FAILURE);
    }

    if (__builtin_bswap32(magic_number) != 2049) {
        printf("Error: Unrecognisable file type %s\n", path_to_labels);
        exit(EXIT_FAILURE);
    }

    // find the size of the file, and go to the end of the header
    fseek(file, 0, SEEK_END);
    size_of_file = ftell(file) - 16;
    fseek(file, 8, SEEK_SET);

    // read the file into a buffer
    buffer_file = malloc(size_of_file * sizeof(unsigned char));
    if (buffer_file == NULL) {
        printf("Error: Memory allocation failed during preprocessing\n");
        exit(EXIT_FAILURE);
    }

    fread(buffer_file, 1, NUMBER_OF_EXAMPLES, file);

    // one-hot encode the training labels
    for (int set_index = 0; set_index < NUMBER_OF_EXAMPLES; set_index++) {
        int label = buffer_file[set_index];

        for (int label_index = 0; label_index < NUMBER_OF_POSSIBILITIES; label_index++) {
            s->labels[set_index * NUMBER_OF_POSSIBILITIES + label_index] = (label_index == label) ? 1 : 0;
        }
    }

    // deallocate memory from buffer_file, set the pointer to NULL, and close file
    free(buffer_file);
    buffer_file = NULL;

    fclose(file);
}

// ----- initialize the layers in the neural network  -----
void initialize_with_random(float input_matrix[], int length) {
    for (int i = 0; i < length; i++) {
        input_matrix[i] = (rand() / (float)RAND_MAX) * 0.2 - 0.1;
    }
}

void initialize_layers(struct parameters *s) {
    initialize_with_random(s->weights_1, HIDDEN_LAYER_1_SIZE * 784);
    memset(s->biases_1, 0, HIDDEN_LAYER_1_SIZE * sizeof(float)); // initialize with zero

    initialize_with_random(s->weights_2, HIDDEN_LAYER_2_SIZE * HIDDEN_LAYER_1_SIZE);
    memset(s->biases_2, 0, HIDDEN_LAYER_2_SIZE * sizeof(float)); // initialize with zero

    initialize_with_random(s->weights_output, NUMBER_OF_POSSIBILITIES * HIDDEN_LAYER_2_SIZE);
    memset(s->biases_output, 0, NUMBER_OF_POSSIBILITIES * sizeof(float)); // initialize with zero
}

// --- fisher-yates shuffle the dataset ---
void shuffle(struct dataset *s) {
    int indices[NUMBER_OF_EXAMPLES];
    float *buffer_images;
    int *buffer_labels;

    // initialize indices with 0, 1, 2, 3...
    for (int i = 0; i < NUMBER_OF_EXAMPLES; i++) {
        indices[i] = i;
    }

    // shuffle indices
    for (int i = 0; i < NUMBER_OF_EXAMPLES; i++) {
        int j = i + rand() / (RAND_MAX / (NUMBER_OF_EXAMPLES - i) + 1);
        int t = indices[j];
        indices[j] = indices[i];
        indices[i] = t;
    }

    // apply the shuffled indices to the images and labels, whilst copying the images and labels to buffers
    buffer_images = malloc(NUMBER_OF_EXAMPLES * IMAGE_SIZE * sizeof(float));
    buffer_labels = malloc(NUMBER_OF_EXAMPLES * NUMBER_OF_POSSIBILITIES * sizeof(int));
    if (buffer_images == NULL || buffer_labels == NULL) {
        printf("Error: Memory allocation failed during shuffling\n");
        exit(EXIT_FAILURE);
    }

    for (int i = 0; i < NUMBER_OF_EXAMPLES; i++) {
        memcpy(buffer_images + i * IMAGE_SIZE, s->images + indices[i] * IMAGE_SIZE, IMAGE_SIZE * sizeof(float));
        memcpy(buffer_labels + i * NUMBER_OF_POSSIBILITIES, s->labels + indices[i] * NUMBER_OF_POSSIBILITIES, NUMBER_OF_POSSIBILITIES * sizeof(int));
    }

    // copy the images and labels back from the bufffers
    memcpy(s->images, buffer_images, NUMBER_OF_EXAMPLES * IMAGE_SIZE * sizeof(float));
    memcpy(s->labels, buffer_labels, NUMBER_OF_EXAMPLES * NUMBER_OF_POSSIBILITIES * sizeof(int));

    // deallocate memory from buffer_images and buffer_labels, and set pointers to NULL
    free(buffer_images);
    buffer_images = NULL;

    free(buffer_labels);
    buffer_labels = NULL;
}

// --- forward pass using sigmoid and softmax ---
void weighted_sum(float input_vector[], float weights[], float biases[], float output_vector[], int number_of_rows, int number_of_columns) {
    // multiply a vector with a matrix
    for (int row_index = 0; row_index < number_of_rows; row_index++) {
            output_vector[row_index] = 0;
            for (int column_index = 0; column_index < number_of_columns; column_index++) {
                output_vector[row_index] += weights[row_index * number_of_columns + column_index] * input_vector[column_index];
        }
    }

    // add two vectors
    for (int i = 0; i < number_of_rows; i++) {
        output_vector[i] += biases[i];
    }
}

void sigmoid(float input_vector[], float output_vector[], int length) {
    for (int i = 0; i < length; i++) {
        output_vector[i] = 1.0 / (1.0 + exp(-input_vector[i]));
    }
}

void softmax(float input_vector[], float output_vector[], int length) {
    float buffer_sum = 0.0;

    for (int i = 0; i < length; i++) {
        buffer_sum += exp(input_vector[i]);
    }

    for (int i = 0; i < length; i++) {
        output_vector[i] = exp(input_vector[i]) / buffer_sum;
    }
}

void forward_pass(float input_images[], struct parameters *s, float output_probabilities[]) {
    float buffer_image[IMAGE_SIZE];

    for (int i = 0; i < BATCH_SIZE; i++) {
        // move image into buffer
        memcpy(buffer_image, &input_images[i * IMAGE_SIZE], IMAGE_SIZE * sizeof(float));

        // first hidden layer
        float pre_activation_1[HIDDEN_LAYER_1_SIZE];
        float post_activation_1[HIDDEN_LAYER_1_SIZE];
        weighted_sum(buffer_image, s->weights_1, s->biases_1, pre_activation_1, HIDDEN_LAYER_1_SIZE, IMAGE_SIZE);
        sigmoid(pre_activation_1, post_activation_1, HIDDEN_LAYER_1_SIZE);

        // second hidden layer
        float pre_activation_2[HIDDEN_LAYER_2_SIZE];
        float post_activation_2[HIDDEN_LAYER_2_SIZE];
        weighted_sum(post_activation_1, s->weights_2, s->biases_2, pre_activation_2, HIDDEN_LAYER_2_SIZE, HIDDEN_LAYER_1_SIZE);
        sigmoid(pre_activation_2, post_activation_2, HIDDEN_LAYER_2_SIZE);

        // output layer
        float pre_activation_output[NUMBER_OF_POSSIBILITIES];
        float post_activation_output[NUMBER_OF_POSSIBILITIES];
        weighted_sum(post_activation_2, s->weights_output, s->biases_output, pre_activation_output, NUMBER_OF_POSSIBILITIES, HIDDEN_LAYER_2_SIZE);
        softmax(pre_activation_output, post_activation_output, NUMBER_OF_POSSIBILITIES);

        // move post_activation_output into output_probabilities
        memcpy(&output_probabilities[i * NUMBER_OF_POSSIBILITIES], post_activation_output, NUMBER_OF_POSSIBILITIES * sizeof(float));
    }
}

// --- compute loss using cross entropy loss (not needed for softmax/sgd) ---
void compute_loss(float input_probabilities[], int input_labels[], float *output_loss) {
    *output_loss = 0.0;

    int index = BATCH_SIZE * NUMBER_OF_POSSIBILITIES;
    for (int i = 0; i < index; i++){
        if (input_probabilities[i] > 0) { // make sure probability > 0 to avoid log(0)
            *output_loss -= log(input_probabilities[i]) * input_labels[i];
        }
    }
}

// --- backpropogation ---
void backpropogation(float input_probabilities[], int input_labels[], float input_images[], struct parameters *l, struct gradients *g) {
    float output_layer_error[BATCH_SIZE * NUMBER_OF_POSSIBILITIES];
    float hidden_layer_2_error[BATCH_SIZE * HIDDEN_LAYER_2_SIZE];
    float hidden_layer_1_error[BATCH_SIZE * HIDDEN_LAYER_1_SIZE];

    // initialize gradients with zero
    memset(g, 0, sizeof(struct gradients));

    // calculate error of output layer (input_probabilities - input_labels)
    for (int batch_index = 0; batch_index < BATCH_SIZE; batch_index++) {
        for (int node_index = 0; node_index < NUMBER_OF_POSSIBILITIES; node_index++) {
            int index = batch_index * NUMBER_OF_POSSIBILITIES + node_index;
            output_layer_error[index] = input_probabilities[index] - input_labels[index];
        }
    }

    // calculate gradients for output layer weights (product(ouput layer error, activation from hidden layer 2)) and biases (output layer error)
    for (int batch_index = 0; batch_index < BATCH_SIZE; batch_index++) {
        for (int node_index = 0; node_index < NUMBER_OF_POSSIBILITIES; node_index++) {
            for (int prev_node_index = 0; prev_node_index < HIDDEN_LAYER_2_SIZE; prev_node_index++) {
                int weight_index = node_index * HIDDEN_LAYER_2_SIZE + prev_node_index;
                int output_index = batch_index * NUMBER_OF_POSSIBILITIES + node_index;
                int hidden_index = batch_index * HIDDEN_LAYER_2_SIZE + prev_node_index;
                g->delta_weights_output[weight_index] += output_layer_error[output_index] * input_probabilities[hidden_index];
            }

            g->delta_biases_output[node_index] += output_layer_error[batch_index * NUMBER_OF_POSSIBILITIES + node_index];
        }
    }

    // calculate error of second hidden layer ((product(transpose(l->weights_output), output layer error) * derivative(activation from hidden layer 2))
    for (int batch_index = 0; batch_index < BATCH_SIZE; batch_index++) {
        for (int node_index = 0; node_index < HIDDEN_LAYER_2_SIZE; node_index++) {
            hidden_layer_2_error[batch_index * HIDDEN_LAYER_2_SIZE + node_index] = 0;

            for (int prev_node_index = 0; prev_node_index < NUMBER_OF_POSSIBILITIES; prev_node_index++) {
                int weight_index = prev_node_index * HIDDEN_LAYER_2_SIZE + node_index;
                int error_index = batch_index * NUMBER_OF_POSSIBILITIES + prev_node_index;
                hidden_layer_2_error[batch_index * HIDDEN_LAYER_2_SIZE + node_index] += output_layer_error[error_index] * l->weights_output[weight_index];
            }

            int hidden_index = batch_index * HIDDEN_LAYER_2_SIZE + node_index;
            hidden_layer_2_error[batch_index * HIDDEN_LAYER_2_SIZE + node_index] *= input_probabilities[hidden_index] * (1 - input_probabilities[hidden_index]);
        }
    }

    // calculate gradients for second hidden layer weights (product(second hidden layer error, activations from hidden layer 1)) and biases (second hidden layer error)
    for (int batch_index = 0; batch_index < BATCH_SIZE; batch_index++) {
        for (int node_index = 0; node_index < HIDDEN_LAYER_2_SIZE; node_index++) {
            for (int prev_node_index = 0; prev_node_index < HIDDEN_LAYER_1_SIZE; prev_node_index++) {
                int weight_index = node_index * HIDDEN_LAYER_1_SIZE + prev_node_index;
                int hidden_index = batch_index * HIDDEN_LAYER_2_SIZE + node_index;
                int activation_index = batch_index * HIDDEN_LAYER_1_SIZE + prev_node_index;
                g->delta_weights_2[weight_index] += hidden_layer_2_error[hidden_index] * input_probabilities[activation_index];
            }

            g->delta_biases_2[node_index] += hidden_layer_2_error[batch_index * HIDDEN_LAYER_2_SIZE + node_index];
        }
    }

    // calculate error of first hidden layer ((product(transpose(l->weights_2)) second hidden layer error) * derivative(activation from hidden layer 1))
    for (int batch_index = 0; batch_index < BATCH_SIZE; batch_index++) {
        for (int node_index = 0; node_index < HIDDEN_LAYER_1_SIZE; node_index++) {
            hidden_layer_1_error[batch_index * HIDDEN_LAYER_1_SIZE + node_index] = 0;

            for (int prev_node_index = 0; prev_node_index < HIDDEN_LAYER_2_SIZE; prev_node_index++) {
                int weight_index = prev_node_index * HIDDEN_LAYER_1_SIZE + node_index;
                int error_index = batch_index * HIDDEN_LAYER_2_SIZE + prev_node_index;
                hidden_layer_1_error[batch_index * HIDDEN_LAYER_1_SIZE + node_index] += hidden_layer_2_error[error_index] * l->weights_2[weight_index];
            }

            int hidden_index = batch_index * HIDDEN_LAYER_1_SIZE + node_index;
            hidden_layer_1_error[batch_index * HIDDEN_LAYER_1_SIZE + node_index] *= input_probabilities[hidden_index] * (1 - input_probabilities[hidden_index]);
        }
    }

    // calculate gradients for first hidden layer weights (product(first hidden layer error, input_images)) and biases (first hidden layer error)
    for (int batch_index = 0; batch_index < BATCH_SIZE; batch_index++) {
        for (int node_index = 0; node_index < HIDDEN_LAYER_1_SIZE; node_index++) {
            for (int prev_node_index = 0; prev_node_index < IMAGE_SIZE; prev_node_index++) {
                int weight_index = node_index * IMAGE_SIZE + prev_node_index;
                int hidden_index = batch_index * HIDDEN_LAYER_1_SIZE + node_index;
                int input_index = batch_index * IMAGE_SIZE + prev_node_index;
                g->delta_weights_1[weight_index] += hidden_layer_1_error[hidden_index] * input_images[input_index];
            }

            g->delta_biases_1[node_index] += hidden_layer_1_error[batch_index * HIDDEN_LAYER_1_SIZE + node_index];
        }
    }
}

// --- update parameters using stochastic gradient descent ---
void update_parameters(struct parameters *l, struct gradients *g) {
    int index;

    // update weights and biases for first hidden layer
    index = HIDDEN_LAYER_1_SIZE * IMAGE_SIZE;
    for (int i = 0; i < index; i++) {
        l->weights_1[i] -= LEARNING_RATE * g->delta_weights_1[i];
    }

    for (int i = 0; i < HIDDEN_LAYER_1_SIZE; i++) {
        l->biases_1[i] -= LEARNING_RATE * g->delta_biases_1[i];
    }

    // update weights and biases for second hidden layer
    index = HIDDEN_LAYER_2_SIZE * HIDDEN_LAYER_1_SIZE;
    for (int i = 0; i < index; i++) {
        l->weights_2[i] -= LEARNING_RATE * g->delta_weights_2[i];
    }

    for (int i = 0; i < HIDDEN_LAYER_2_SIZE; i++) {
        l->biases_2[i] -= LEARNING_RATE * g->delta_biases_2[i];
    }

    // update weights and biases for output layer
    index = NUMBER_OF_POSSIBILITIES * HIDDEN_LAYER_2_SIZE;
    for (int i = 0; i < index; i++) {
        l->weights_output[i] -= LEARNING_RATE * g->delta_weights_output[i];
    }
}

// --- output the weights and biases to a file ---
void output_parameters(struct parameters *s) {
    FILE *file;

    file = fopen("parameters.bin", "wb");
    if (file == NULL) {
        printf("Error: Unable to create output file\n");
        exit(EXIT_FAILURE);
    }

    if (fwrite(s, sizeof(struct parameters), 1, file) != 1) {
        printf("Error: Unable to write struct parameters to file\n");
        exit(EXIT_FAILURE);
    }

    fclose(file);
}

// --- main ---
int main(int argc, char *argv[])
{
    // error handling for command-line arguments
    if (argc < 4) {
        printf("Usage: %s <images_file_path> <labels_file_path> <epochs>\n", argv[0]);
        exit(EXIT_FAILURE);
    }

    // seed random number generator with current time
    srand(time(NULL));

    // allocate memory to structs that will store the dataset, weights, biases, and gradients
    struct dataset *dataset = malloc(sizeof(struct dataset));
    struct parameters *parameters = malloc(sizeof(struct parameters));
    struct gradients *gradients = malloc(sizeof(struct gradients));
    if (dataset == NULL || parameters == NULL || gradients == NULL) {
        printf("Error: Memory allocation failed before preprocessing\n");
        exit(EXIT_FAILURE);
    }

    // buffers
    float batch_images[BATCH_SIZE * IMAGE_SIZE];
    int batch_labels[BATCH_SIZE * NUMBER_OF_POSSIBILITIES];
    float batch_probabilities[BATCH_SIZE * NUMBER_OF_POSSIBILITIES];
    float batch_loss;

    // preprocess the dataset
    printf("Preprocessing images...\n");
    preprocess_images(argv[1], dataset);
    printf("Images preprocessed\n");
    printf("Preprocessing labels...\n");
    preprocess_labels(argv[2], dataset);
    printf("Labels preprocessed\n");

    // initialize weights with small random numbers, and biases with zero
    printf("Initializing parameters...\n");
    initialize_layers(parameters);
    printf("Parameters initialized\n");

    // loop for number of epochs
    printf("Starting training...\n");

    int epochs = atoi(argv[3]);
    for (int i = 0; i < epochs; i++) {
        // shuffle the dataset
        shuffle(dataset);

        // loop for number of batches
        int number_of_batches = NUMBER_OF_EXAMPLES /  BATCH_SIZE;
        for (int j = 0; j < number_of_batches; j++) {
            // extract a batch from the dataset
            memcpy(batch_images, &dataset->images[j * BATCH_SIZE * IMAGE_SIZE], BATCH_SIZE * IMAGE_SIZE * sizeof(float));
            memcpy(batch_labels, &dataset->labels[j * BATCH_SIZE * NUMBER_OF_POSSIBILITIES], BATCH_SIZE * NUMBER_OF_POSSIBILITIES * sizeof(int));

            // iteration
            forward_pass(batch_images, parameters, batch_probabilities);
            compute_loss(batch_probabilities, batch_labels, &batch_loss); // not needed for softmax/sgd, just for logging
            backpropogation(batch_probabilities, batch_labels, batch_images, parameters, gradients);
            update_parameters(parameters, gradients);

            // logging
            printf("Epoch: %d, Batch: %d, Loss: %.4f\n", i, j, batch_loss);
        }
    }

    // output the weights and biases to a file
    printf("Writing parameters to file...\n");
    output_parameters(parameters);
    printf("Parameters successfully written to file parameters.bin\n");

    // deallocate memory from structs that stored the dataset, weights, biases, and gradients, and set pointers to NULL
    free(dataset);
    dataset = NULL;
    free(parameters);
    parameters = NULL;
    free(gradients);
    gradients = NULL;

    printf("Training complete.\n");
    return 0;
}
