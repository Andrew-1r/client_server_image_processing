#include <endian.h>  // or <byteswap.h> / <sys/endian.h> as appropriate
#include <fcntl.h>
#include <netdb.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/sendfile.h>
#include <sys/stat.h>
#include <unistd.h>

const char charExample = '>';
const char *const outputImageString = "--outputimage";
const char *const replaceImageString = "--replaceimage";
const char *const detectImageString = "--detectfile";
const char *const usageError =
    "Usage: ./uqfaceclient port [--outputimage filename] [--replaceimage "
    "filename] [--detectfile filename]\n";
const char *const fileError =
    "uqfaceclient: unable to open the input file \"FILENAME HERE\" for "
    "reading\n";
const char *const communicationError =
    "uqfaceclient: unexpected communication error\n";

#define OPERATION_FACE_DETECTION 0
#define OPERATION_FACE_REPLACEMENT 1
#define OPERATION_OUTPUT_IMAGE 2
#define OPERATION_ERROR_MESSAGE 3
#define USAGE_ERROR_EXIT 6
#define INVALID_INPUT_FILE_EXIT 11
#define INVALID_OUTPUT_FILE_EXIT 9
#define INVALID_PORT_EXIT 1
#define COMMUNICATION_EXIT 16
#define ERROR_MESSAGE_EXIT 5
#define READ_WRITE_OWNER_GROUP_OTHER_PERMISSION 0666
#define FOUR_KB_CHUNK 4096
#define PREFIX 0x23107231

// Struct to store command line arguments
struct Args {
  char *port;
  bool detectImagePresent;
  char *detectImage;
  bool outputImagePresent;
  char *outputImage;
  bool replaceImagePresent;
  char *replaceImage;
};

// Frees all dynamically-allocated members inside an Args struct
// Inputs:  args – pointer to dynamically-allocated Args
// Returns: void (releases memory)
void free_args(struct Args *args) {
  free(args->port);
  free(args->detectImage);
  free(args->outputImage);
  free(args->replaceImage);
  free(args);
}

// Checks whether the command line omitted the port entirely
// Inputs:  argc – argument count
// Returns: true  if argc < 2
//          false otherwise
bool no_port_present(int argc) {
  if (argc < 2) {
    return true;
  }

  return false;
}

// Determines if the sole argument provided is the port number
// Inputs:  argc – argument count
// Returns: true  if argc == 2
//          false otherwise
bool only_arg_is_port(int argc) {
  if (argc == 2) {
    return true;
  }
  return false;
}

// Detects any empty strings among the command-line arguments
// Inputs:  argc – argument count
//          argv – argument vector
// Returns: true  if at least one arg is empty or NULL
//          false otherwise
bool empty_string_present(int argc, char **argv) {
  for (int i = 1; i < argc; i++) {
    if (argv[i] == NULL || argv[i][0] == '\0') {
      return true;
    }
  }
  return false;
}

// Tests whether a string exactly matches any valid optional flag
// Inputs:  string – candidate argument
// Returns: true  if it equals --outputimage/--detectfile/--replaceimage
//          false otherwise
bool string_is_an_optional_arg(char *string) {
  if (strcmp(string, outputImageString) == 0) {
    return true;
  }
  if (strcmp(string, detectImageString) == 0) {
    return true;
  }
  if (strcmp(string, replaceImageString) == 0) {
    return true;
  }
  return false;
}

// Validates that every optional flag is followed by a filename
// Inputs:  argc / argv – raw command line
// Returns: true  if a flag appears as the final token
//          false otherwise
bool option_present_without_filename(int argc, char **argv) {
  for (int i = 1; i < argc; i++) {
    // if an option is found, if nothing follows it, return true
    if ((string_is_an_optional_arg(argv[i])) && i == (argc - 1)) {
      return true;
    }
  }
  return false;
}

// Detects repeated appearances of the same optional flag
// Inputs:  argc / argv – raw command line
// Returns: true  if any flag appears more than once
//          false otherwise
bool duplicate_args(int argc, char **argv) {
  bool outputImagePresent = false;
  bool detectImagePresent = false;
  bool replaceImagePresent = false;

  for (int i = 1; i < argc; i++) {
    if (strcmp(argv[i], outputImageString) == 0) {
      if (outputImagePresent) {
        return true;
      }
      outputImagePresent = true;
    } else if (strcmp(argv[i], detectImageString) == 0) {
      if (detectImagePresent) {
        return true;
      }
      detectImagePresent = true;
    } else if (strcmp(argv[i], replaceImageString) == 0) {
      if (replaceImagePresent) {
        return true;
      }
      replaceImagePresent = true;
    }
  }
  return false;
}

// Flags any unexpected or malformed arguments after the port
// Inputs:  argc / argv – raw command line
// Returns: true  if an unknown token is encountered
//          false otherwise
bool unexpected_args_found(int argc, char **argv) {
  // start i at 2 to skip port, skip past optional arg filenames
  for (int i = 2; i < argc; i++) {
    if (string_is_an_optional_arg(argv[i])) {
      i++;
    } else {
      return true;
    }
  }
  return false;
}

// Orchestrates all client-side CLI validation helpers
// Inputs:  argc / argv – raw command line
// Returns: true  if any validation fails, false otherwise
bool invalid_command_line_args(int argc, char **argv) {
  if (no_port_present(argc)) {
    return true;
  }

  if (empty_string_present(argc, argv)) {
    return true;
  }

  if (only_arg_is_port(argc)) {
    return false;
  }

  if (option_present_without_filename(argc, argv)) {
    return true;
  }

  if (duplicate_args(argc, argv)) {
    return true;
  }

  if (unexpected_args_found(argc, argv)) {
    return true;
  }

  return false;
}

// Allocates and fills an Args struct based on valid CLI arguments
// Inputs:  argc / argv – raw command line
//          args       – pointer to calloc’d Args to populate
// Returns: void (strings duplicated where necessary)
void populate_args_struct(int argc, char **argv, struct Args *args) {
  args->port = strdup(argv[1]);

  for (int i = 2; i < argc; i++) {
    if (strcmp(argv[i], outputImageString) == 0) {
      args->outputImagePresent = true;
      args->outputImage = strdup(argv[++i]);
    } else if (strcmp(argv[i], detectImageString) == 0) {
      args->detectImagePresent = true;
      args->detectImage = strdup(argv[++i]);
    } else if (strcmp(argv[i], replaceImageString) == 0) {
      args->replaceImagePresent = true;
      args->replaceImage = strdup(argv[++i]);
    }
  }
}

// Ensures the replacement image file can be opened for reading
// Inputs:  args – populated Args struct
// Returns: void, but exits with INVALID_INPUT_FILE_EXIT on failure
void validate_replace_file(struct Args *args) {
  FILE *input = fopen(args->replaceImage, "r");

  if (!input) {
    fprintf(stderr,
            "uqfaceclient: unable to open the input file \"%s\" for reading\n",
            args->replaceImage);
    exit(INVALID_INPUT_FILE_EXIT);
  }

  fclose(input);
}

// Ensures the detect image file can be opened for reading
// Inputs:  args – populated Args struct
// Returns: void, but exits with INVALID_INPUT_FILE_EXIT on failure
void validate_input_file(struct Args *args) {
  FILE *input = fopen(args->detectImage, "r");

  if (!input) {
    fprintf(stderr,
            "uqfaceclient: unable to open the input file \"%s\" for reading\n",
            args->detectImage);
    exit(INVALID_INPUT_FILE_EXIT);
  }

  fclose(input);
}

// Ensures the output image file can be opened/created for writing
// Inputs:  args – populated Args struct
// Returns: void, but exits with INVALID_OUTPUT_FILE_EXIT on failure
void validate_output_file(struct Args *args) {
  FILE *output = fopen(args->outputImage, "w");

  if (!output) {
    fprintf(stderr,
            "uqfaceclient: cannot open the output file \"%s\" for writing\n",
            args->outputImage);
    exit(INVALID_OUTPUT_FILE_EXIT);
  }

  fclose(output);
}

// Runs all individual file-permission validators as required
// Inputs:  args – populated Args struct
// Returns: void (exits on any validation error)
void validate_files(struct Args *args) {
  if (args->outputImagePresent) {
    validate_output_file(args);
  }
  if (args->detectImagePresent) {
    validate_input_file(args);
  }
  if (args->replaceImagePresent) {
    validate_replace_file(args);
  }
}

// Reads raw image data from stdin, buffers it, and sends to server
// Inputs:  socketFd – connected socket to server
// Returns: void (writes size prefix + binary payload)
void no_detect_image_present_helper(int socketFd) {
  // Read raw bytes from stdin
  size_t bufferCapacity = 0;     // current size of buffer
  size_t bufferLength = 0;       // actual number of bytes in buffer
  uint8_t *buffer = NULL;        // pointer to full buffer of stdin data
  uint8_t chunk[FOUR_KB_CHUNK];  // size of stdin to read at a time
  ssize_t chunkBytes;            // number of bytes read in last read call

  // slurp stdin into buffer
  while ((chunkBytes = read(STDIN_FILENO, chunk, sizeof chunk)) > 0) {
    // make buffer bigger based on current capacity and size of chunk
    // of data coming in
    if (bufferLength + chunkBytes > bufferCapacity) {
      bufferCapacity = bufferCapacity * 2 + chunkBytes;
      buffer = realloc(buffer, bufferCapacity);
    }
    // memcpy() concatonates data from chunk into the buffer
    memcpy(buffer + bufferLength, chunk, chunkBytes);
    bufferLength += chunkBytes;
  }

  // exit on error reading from stdin, not sure what message to put here
  if (chunkBytes < 0) {
    perror("uqfaceclient: error reading from stdin");
    free(buffer);
    exit(INVALID_INPUT_FILE_EXIT);
  }

  // change the size of the data thats about to be sent to little endian
  // and send this to server
  uint32_t stdinSizeLe = htole32((uint32_t)bufferLength);
  write(socketFd, &stdinSizeLe, sizeof stdinSizeLe);

  // send the actual binary data to server
  write(socketFd, buffer, bufferLength);
  free(buffer);
}

// Handles sending the detect image (either file or stdin) to server
// Inputs:  socketFd – connected socket
//          args     – populated Args struct
// Returns: void (writes size prefix + image)
// helper function to handle the detect image given to client
void detect_image_helper(int socketFd, const struct Args *args) {
  if (args->detectImagePresent) {
    // open detect image file descriptor
    int detectImageFd = open(args->detectImage, O_RDONLY);

    // make struct with st_size, which is total size in bytes
    struct stat detectImageStat;
    fstat(detectImageFd, &detectImageStat);

    // convert the size into little endian and send to server
    uint32_t detectImageSizeLe = htole32((uint32_t)detectImageStat.st_size);
    write(socketFd, &detectImageSizeLe, sizeof detectImageSizeLe);

    // send the detect image through to the server
    off_t offset = 0;
    sendfile(socketFd, detectImageFd, &offset, detectImageStat.st_size);

    close(detectImageFd);
  }
  // if no detect image, read raw bytes from stdin
  else {
    no_detect_image_present_helper(socketFd);
  }
}

// Constructs and transmits an entire request to the server
// Inputs:  socketFd – connected socket
//          args     – populated Args struct
// Returns: void (writes prefix, op code, images, etc.)
void send_request(int socketFd, struct Args *args) {
  // make prefix little endian and send to server
  uint32_t prefixLe = htole32(PREFIX);

  // send prefix to server
  write(socketFd, &prefixLe, sizeof prefixLe);

  // determine and send operation, 1 if replacing faces, 0 if detecting faces
  uint8_t operationType = args->replaceImagePresent ? OPERATION_FACE_REPLACEMENT
                                                    : OPERATION_FACE_DETECTION;
  write(socketFd, &operationType, sizeof operationType);

  detect_image_helper(socketFd, args);

  /* if replacement requested, open and send the replace image */
  if (args->replaceImagePresent) {
    // open fd for replace image
    int replaceImageFd = open(args->replaceImage, O_RDONLY);

    // create a populate struct with st_size with size in bytes
    struct stat replaceImageStat;
    fstat(replaceImageFd, &replaceImageStat);

    // convert size to little endian and send to server
    uint32_t replaceSizeLe = htole32((uint32_t)replaceImageStat.st_size);
    write(socketFd, &replaceSizeLe, sizeof replaceSizeLe);

    // send the replace image file through to the server
    off_t offset = 0;
    sendfile(socketFd, replaceImageFd, &offset, replaceImageStat.st_size);

    close(replaceImageFd);
  }
}

// Reads exactly n bytes from fd into buf (blocking)
// Inputs:  fd  – file descriptor
//          buf – destination buffer
//          n   – required byte count
// Returns: 0 on success, –1 on EOF / error
int read_exact(int fd, void *buf, size_t n) {
  size_t count = n;
  unsigned char *buffer = buf;

  // read count characters from fd, write them to buffer
  while (count > 0) {
    ssize_t bytesRead = read(fd, buffer, count);

    // exit if we unexpectadely run off the end of the fd
    if (bytesRead <= 0)  // 0 = EOF, <0 = error
    {
      return -1;
    }

    count -= bytesRead;
    buffer += bytesRead;
  }
  return 0;
}

// Reads a 32-bit little-endian unsigned integer from fd
// Inputs:  fd  – file descriptor
//          out – pointer to host-order result
// Returns: 0 on success, –1 on short read / error
int read_u32_le(int fd, uint32_t *out) {
  uint32_t tempLe;

  // read little endian value into a temporary variable
  if (read_exact(fd, &tempLe, sizeof tempLe) < 0) {
    return -1;
  }

  // change it from little endian back to "host byte order"
  *out = le32toh(tempLe);

  return 0;
}

// Prints a generic communication-failure message and exits
// Inputs:  none
// Returns: never returns (exits with COMMUNICATION_EXIT)
void communication_error_exit() {
  fprintf(stderr, "%s", communicationError);
  exit(COMMUNICATION_EXIT);
}

// Either writes an image payload to stdout/file OR prints a server error
// Inputs:  operationType – OPERATION_OUTPUT_IMAGE or OPERATION_ERROR_MESSAGE
//          payload       – pointer to received bytes
//          payloadLength – byte count (excl. NUL unless error)
//          args          – populated Args struct for output filename choice
// Returns: void (exits on error message)
void output_image_or_error_helper(uint8_t operationType, uint8_t *payload,
                                  uint32_t payloadLength,
                                  const struct Args *args) {
  if (operationType == OPERATION_OUTPUT_IMAGE) {
    // prepare stdout for payload, or output image file if given
    int outputFd = STDOUT_FILENO;
    if (args->outputImagePresent) {
      outputFd = open(args->outputImage, O_CREAT | O_WRONLY | O_TRUNC,
                      READ_WRITE_OWNER_GROUP_OTHER_PERMISSION);
    }

    // write payload to stdout or output image file if given
    write(outputFd, payload, payloadLength);

    // close fd to output image if given
    if (args->outputImagePresent) {
      close(outputFd);
    }
  }
  // print error message to stderr if found
  else if (operationType == OPERATION_ERROR_MESSAGE) {
    payload[payloadLength] = '\0';  // NULL terminate error message
    fprintf(stderr, "uqfaceclient: got the following error message: \"%s\"\n",
            (char *)payload);
    free(payload);
    exit(ERROR_MESSAGE_EXIT);
  }
  // else if operation weren't valid, communication error
  else {
    communication_error_exit();
  }
}

// Consumes the full server response, handling protocol and payload
// Inputs:  socketFd – connected socket
//          args     – populated Args struct (output options)
// TODO: some communication errors should print the error from my server
int read_response(int socketFd, const struct Args *args) {
  // read prefix, exit if incorrect
  uint32_t prefix = 0;
  if (read_u32_le(socketFd, &prefix) < 0 || prefix != PREFIX) {
    communication_error_exit();
  }

  // read operation type should be 2 or 3 from server
  uint8_t operationType = 0;
  if (read_exact(socketFd, &operationType, 1) < 0) {
    communication_error_exit();
  }

  // read how many bytes are in the returned image
  uint32_t payloadLength = 0;
  if (read_u32_le(socketFd, &payloadLength) < 0) {
    communication_error_exit();
  }

  // allocate memory for payload
  // add +1 for NULL terminator if it's an error message
  if (operationType == OPERATION_ERROR_MESSAGE) {
    payloadLength += 1;
  }

  uint8_t *payload = malloc(payloadLength);
  if (payload == NULL) {
    communication_error_exit();
  }

  // write to payload from socket fd
  if (read_exact(socketFd, payload, payloadLength) < 0) {
    free(payload);
    communication_error_exit();
  }

  // send image to file, stdout, or send a received error
  output_image_or_error_helper(operationType, payload, payloadLength, args);

  free(payload);
  return 0;
}

// Program entry point: validates CLI, connects, sends request, handles reply
// Inputs:  argc / argv – raw command-line arguments
// Returns: EXIT_SUCCESS (0) on success, or various error exits
int main(int argc, char **argv) {
  // Validate command line arguments
  if (invalid_command_line_args(argc, argv)) {
    fprintf(stderr, "%s", usageError);
    exit(USAGE_ERROR_EXIT);
  }

  // Create and populate Args struct
  struct Args *args = calloc(1, sizeof(struct Args));
  populate_args_struct(argc, argv, args);

  validate_files(args);

  // create addrinfo struct, assign IPv4 and TCP to hints
  struct addrinfo *ai = 0;
  struct addrinfo hints;
  memset(&hints, 0, sizeof(struct addrinfo));
  hints.ai_family = AF_INET;        // IPv4
  hints.ai_socktype = SOCK_STREAM;  // TCP

  // fill addressinfo ai
  // on error, free ai, print error message, return 1
  int err;
  if ((err = getaddrinfo("localhost", args->port, &hints, &ai))) {
    freeaddrinfo(ai);
    fprintf(stderr, "%s\n", gai_strerror(err));
    return 1;  // could not work out the address
  }

  // opens TCP socket
  int fd = socket(AF_INET, SOCK_STREAM, 0);  // 0 == use default protocol
  // connect()'s socket fd to socket ai.address
  // returns 0 if successful
  if (connect(fd, ai->ai_addr, sizeof(struct sockaddr))) {
    fprintf(stderr,
            "uqfaceclient: unable to connect to the server on port \"%s\"\n",
            args->port);
    exit(INVALID_PORT_EXIT);
  }

  send_request(fd, args);
  int status = read_response(fd, args);

  free_args(args);

  return status;
}