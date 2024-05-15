// --- header files ---
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <time.h>

// --- constants ---
#define NUMBER_OF_EXAMPLES 10000
#define IMAGE_SIZE (28 * 28)
#define NUMBER_OF_POSSIBILITIES 10

#define HIDDEN_LAYER_1_SIZE 512
#define HIDDEN_LAYER_2_SIZE 256

// --- structs ---
struct test_set {
    float images[NUMBER_OF_EXAMPLES * IMAGE_SIZE];
    int labels[NUMBER_OF_EXAMPLES * NUMBER_OF_POSSIBILITIES];
};

struct parameters {
    float weights_1[HIDDEN_LAYER_1_SIZE * IMAGE_SIZE]; // matrix
    float biases_1[HIDDEN_LAYER_1_SIZE]; // vector

    float weights_2[HIDDEN_LAYER_2_SIZE * HIDDEN_LAYER_1_SIZE]; // matrix
    float biases_2[HIDDEN_LAYER_2_SIZE]; // vector

    float weights_output[NUMBER_OF_POSSIBILITIES * HIDDEN_LAYER_2_SIZE]; // matrix
    float biases_output[NUMBER_OF_POSSIBILITIES]; // vector
};

// --- load weights and biases ---
void load_parameters(char path_to_parameters[], struct parameters *s) {
    FILE *file;

    file = fopen(path_to_parameters, "rb");
    if (file == NULL) {
        printf("Error: Unable to open file with parameters\n");
        exit(EXIT_FAILURE);
    }

    if (fread(s, sizeof(struct parameters), 1, file) < 1) {
        printf("Error: Failed to read parameters from file\n");
        exit(EXIT_FAILURE);
    }

    fclose(file);
}

// --- preprocess images and labels ---
void preprocess_images(char path_to_images[], struct test_set *s) {
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

    // normalize pixel values of the test images
    for (int i = 0; i < (NUMBER_OF_EXAMPLES * IMAGE_SIZE); i++) {
        s->images[i] = buffer_file[i] / 255.0;
    }

    free(buffer_file);
    fclose(file);
}

void preprocess_labels(char path_to_labels[], struct test_set *s) {
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

// --- forward pass using sigmoid and softmax ---
void weighted_sum(float input_vector[], float weights[], float biases[], float output_vector[], int rows, int columns) {
    // vector matrix multiply
    for (int i = 0; i < rows; i++) {
            output_vector[i] = 0;
            for (int j = 0; j < columns; j++) {
                output_vector[i] += weights[i * columns + j] * input_vector[j];
        }
    }

    // vector add
    for (int i = 0; i < rows; i++) {
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

    for (int i = 0; i < NUMBER_OF_EXAMPLES; i++) {
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

// --- compare predicted probability with labels ---
void compare(float input_probabilities[], int input_labels[], int output_correct[]) {
    int example_index, class_index, max_index;

    // loop for number of examples
    for (example_index = 0; example_index < NUMBER_OF_EXAMPLES; example_index++) {
        float max_probability = -1.0;
        max_index = 0;

        // find the class with the highest probability for the current example
        for (class_index = 0; class_index < NUMBER_OF_POSSIBILITIES; class_index++) {
            int index = example_index * NUMBER_OF_POSSIBILITIES + class_index;
            if (input_probabilities[index] > max_probability) {
                max_probability = input_probabilities[index];
                max_index = class_index;
            }
        }

        // check if the class with the highest probability matches the one-hot encoded label
        int label_index = example_index * NUMBER_OF_POSSIBILITIES + max_index;
        output_correct[example_index] = input_labels[label_index] == 1 ? 1 : 0;
    }
}

int percentage(int input_correct[]) {
    int count = 0;

    // count number of correct guesses
    for (int i = 0; i < NUMBER_OF_EXAMPLES; i++) {
        if (input_correct[i] == 1) {
            count++;
        }
    }

    // return number as a percentage
    return (count * 100) / NUMBER_OF_EXAMPLES;
}

// --- main ---
int main(int argc, char *argv[])
{
    // error handling for command-line arguments
    if (argc < 4) {
        printf("Usage: %s <parameters_file_path> <images_file_path> <labels_file_path>\n", argv[0]);
        exit(EXIT_FAILURE);
    }

    // seed random number generator with current time
    srand(time(NULL));

    // allocate memory to struct that will store the test set, weights, and biases
    struct test_set *test_set = malloc(sizeof(struct test_set));
    struct parameters *parameters = malloc(sizeof(struct parameters));
    if (test_set == NULL || parameters == NULL) {
        printf("Error: Memory allocation failed before preprocessing\n");
        exit(EXIT_FAILURE);
    }

    // buffers
    float probability[NUMBER_OF_EXAMPLES * NUMBER_OF_POSSIBILITIES];
    int correct[NUMBER_OF_EXAMPLES];

    // load weights and biases
    load_parameters(argv[1], parameters);

    // preprocess images and labels
    preprocess_images(argv[2], test_set);
    preprocess_labels(argv[3], test_set);

    // forward pass
    forward_pass(test_set->images, parameters, probability);

    // compare images with labels
    compare(probability, test_set->labels, correct);

    // find and print percentage that is correct
    printf("Percentage of correct guesses: %d%%\n", percentage(correct));

    // deallocate memory from struct that stored the test set, weights, and biases
    free(test_set);
    test_set = NULL;
    free(parameters);
    parameters = NULL;

    return 0;
}
