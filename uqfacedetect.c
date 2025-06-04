#include <arpa/inet.h>
#include <ctype.h>
#include <fcntl.h>
#include <netdb.h>
#include <netinet/in.h>
#include <opencv/cv.h>
#include <opencv2/imgcodecs/imgcodecs_c.h>
#include <opencv2/imgproc/imgproc_c.h>
#include <opencv2/objdetect/objdetect_c.h>
#include <pthread.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/file.h>
#include <sys/sendfile.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

const char *const usageError =
    "Usage: ./uqfacedetect clientlimit maxsize [portnumber]\n";
const char *const ephemeral = "0";
const char *const tempImage = "/tmp/imagefile.jpg";
const char *const tempImageError =
    "uqfacedetect: unable to open the image file for writing\n";

const char *const faceCascadeFilename =
    "/local/courses/csse2310/resources/a4/haarcascade_frontalface_alt2.xml";
const char *const eyesCascadeFilename =
    "/local/courses/csse2310/resources/a4/haarcascade_eye_tree_eyeglasses.xml";
const char *const cascadeClassifierError =
    "uqfacedetect: unable to load a cascade classifier\n";
const char *const responseFile =
    "/local/courses/csse2310/resources/a4/responsefile";
const char *const loadReplaceImage = "loadReplaceImage";
const char *const loadDetectImage = "loadDetectImage";

const char *const invalidProtocolError = "invalid message";
const char *const invalidOperationError = "invalid operation type";
const char *const zeroByteImageError = "image is 0 bytes";
const char *const imageTooLargeError = "image too large";
const char *const imageLoadFailureError = "invalid image";
const char *const noFacesDetectedError = "no faces detected in image";

// OpenCV parameters
const float haarScaleFactor = 1.1;
const int haarMinNeighbours = 4;
const int haarFlags = 0;
const int haarMinSize = 0;
const int haarMaxSize = 1000;
const int ellipseStartAngle = 0;
const int ellipseEndAngle = 360;
const int lineThickness = 4;
const int lineType = 8;
const int shift = 0;
const int bgraChannels = 4;
const int alphaIndex = 3;

#define PREFIX 0x23107231
#define OPERATION_FACE_DETECTION 0
#define OPERATION_FACE_REPLACEMENT 1
#define OPERATION_OUTPUT_IMAGE 2
#define OPERATION_ERROR_MESSAGE 3

#define USAGE_ERROR_EXIT 6
#define UNABLE_TO_LISTEN_EXIT 19
#define TEMP_IMAGE_EXIT 9
#define CASCADE_CLASSIFIER_EXIT 14

#define CLIENT_LIMIT 1
#define MAX_SIZE 2
#define PORT_NUMBER 3
#define CLIENT_LIMIT_AND_MAX_SIZE_GIVEN 3
#define OPTIONAL_PORT_GIVEN 4
#define FOUR_KB_CHUNK 4096
#define READ_WRITE_OWNER_PERMISSION 0600

#define BASE 10
#define CLIENT_LIMIT_MAX 10000
#define MAX_SIZE_MAX UINT32_MAX

// Struct to store command line arguments
struct Args {
  int clientLimit;
  // https://stackoverflow.com/questions/1229131/how-to-declare-a-32-bit-integer-in-c
  uint32_t maxSize;
  char *port;
};

// Struct to store arguments for threads
struct ThreadArgs {
  uint32_t maxSize;
  int fd;
  pthread_mutex_t *lock;
};

// Frees all dynamically-allocated members of an Args struct
// Inputs:  args – pointer to dynamically-allocated Args
// Returns: void (memory for args and its members is released)
void free_args(struct Args *args) {
  free(args->port);
  free(args);
}

// Populates an Args struct from the command-line arguments
// Inputs:  argc – argument count
//          argv – argument vector
//          args – pointer to already-calloc’d Args to fill
// Returns: void (fields inside *args are set; strings are strdup’d)
void populate_args_struct(int argc, char **argv, struct Args *args) {
  int clientLimit = (int)strtol(argv[CLIENT_LIMIT], NULL, BASE);
  uint32_t maxSize = strtoul(argv[MAX_SIZE], NULL, BASE);

  if (clientLimit == 0) {
    // unlimited if clientlimit == 0
    args->clientLimit = INT32_MAX;
  } else {
    args->clientLimit = clientLimit;
  }
  if (maxSize == 0) {
    args->maxSize = MAX_SIZE_MAX;
  } else {
    args->maxSize = maxSize;
  }
  // if no port given, port is ephemeral
  if (argc == CLIENT_LIMIT_AND_MAX_SIZE_GIVEN) {
    args->port = strdup(ephemeral);
  }
  if (argc == OPTIONAL_PORT_GIVEN) {
    args->port = strdup(argv[PORT_NUMBER]);
  }
}

// Verifies argc is either 3 (clientlimit & maxsize) or 4 (plus port)
// Inputs:  argc – argument count
// Returns: true  if argc is invalid,
//          false if argc is exactly 3 or 4
bool invalid_number_of_args(int argc) {
  if (argc != CLIENT_LIMIT_AND_MAX_SIZE_GIVEN && argc != OPTIONAL_PORT_GIVEN) {
    return true;
  }
  return false;
}

// Checks that the client-limit string is a non-negative integer ≤ 10000
// Inputs:  string – string representation of client limit
// Returns: true  if string is invalid (non-integer, negative, > limit)
//          false if string represents a valid client-limit value
bool invalid_client_limit(char *string) {
  // negative number
  if (string[0] == '-') {
    return true;
  }

  char *endPtr;
  long cl = strtol(string, &endPtr, BASE);

  // non int found in string
  if (endPtr == string || endPtr[0] != '\0') {
    return true;
  }

  // client limit out of bounds
  if (cl > CLIENT_LIMIT_MAX) {
    return true;
  }
  return false;
}

// Checks that the max-size string is a non-negative integer ≤ UINT32_MAX
// Inputs:  string – string representation of max image size
// Returns: true  if string is invalid (non-integer, negative, > limit)
//          false if string represents a valid max-size value
bool invalid_max_size(char *string) {
  // negative number
  if (string[0] == '-') {
    return true;
  }

  char *endPtr;
  unsigned long long ms = strtoull(string, &endPtr, BASE);

  // non int found in string
  if (endPtr == string || endPtr[0] != '\0') {
    return true;
  }

  // client limit out of bounds
  if (ms > MAX_SIZE_MAX) {
    return true;
  }
  return false;
}

// Detects any empty command-line strings
// Inputs:  argc – argument count
//          argv – argument vector
// Returns: true  if at least one argument is empty or NULL
//          false otherwise
bool empty_string_present(int argc, char **argv) {
  for (int i = 1; i < argc; i++) {
    if (argv[i] == NULL || argv[i][0] == '\0') {
      return true;
    }
  }
  return false;
}

// Aggregates all argument-validation helpers
// Inputs:  argc – argument count
//          argv – argument vector
// Returns: true  if any argument check fails
//          false if the full command line is valid
bool invalid_command_line_args(int argc, char **argv) {
  if (invalid_number_of_args(argc)) {
    return true;
  }
  if (empty_string_present(argc, argv)) {
    return true;
  }
  if (invalid_client_limit(argv[CLIENT_LIMIT])) {
    return true;
  }
  if (invalid_max_size(argv[MAX_SIZE])) {
    return true;
  }
  return false;
}

// Ensures /tmp/imagefile.jpg exists, is writable and truncatable
// Inputs:  none
// Returns: void, but exits with TEMP_IMAGE_EXIT on failure
void temporary_image_file_check() {
  int tempImageFd =
      open(tempImage, O_CREAT | O_RDWR | O_TRUNC, READ_WRITE_OWNER_PERMISSION);

  if (tempImageFd < 0) {
    fprintf(stderr, "%s", tempImageError);
    exit(TEMP_IMAGE_EXIT);
  }

  close(tempImageFd);
}

// Verifies that the required OpenCV Haar cascades can be loaded
// Inputs:  none
// Returns: void, but exits with CASCADE_CLASSIFIER_EXIT on failure
void cascade_classifier_file_check() {
  CvHaarClassifierCascade *faceCascade =
      (CvHaarClassifierCascade *)cvLoad(faceCascadeFilename, NULL, NULL, NULL);
  CvHaarClassifierCascade *eyeCascade =
      (CvHaarClassifierCascade *)cvLoad(eyesCascadeFilename, NULL, NULL, NULL);

  if (!faceCascade || !eyeCascade) {
    fprintf(stderr, "%s", cascadeClassifierError);
    exit(CASCADE_CLASSIFIER_EXIT);
  }

  cvReleaseHaarClassifierCascade(&faceCascade);
  cvReleaseHaarClassifierCascade(&eyeCascade);
}

// Prints a standard “unable to listen” error for a given port and exits
// Inputs:  port – string used in the error message
// Returns: void (terminates the program via exit)
void unable_to_listen_error(char *port) {
  fprintf(stderr, "uqfacedetect: unable to listen on given port \"%s\"\n",
          port);
  exit(UNABLE_TO_LISTEN_EXIT);
}

// Creates, binds, and listens on a TCP socket (possibly ephemeral)
// Inputs:  args – pointer to validated Args struct
// Returns: listening socket file-descriptor on success
//          (exits if any socket API call fails)
int open_listen(struct Args *args) {
  struct addrinfo *ai = 0;
  struct addrinfo hints;

  memset(&hints, 0, sizeof(struct addrinfo));
  hints.ai_family = AF_INET;  // IPv4
  hints.ai_socktype = SOCK_STREAM;
  hints.ai_flags = AI_PASSIVE;  // listen on all IP addresses

  int err;
  if ((err = getaddrinfo(NULL, args->port, &hints, &ai))) {
    freeaddrinfo(ai);
    unable_to_listen_error(args->port);
  }
  // Create a socket
  int listenfd = socket(AF_INET, SOCK_STREAM, 0);  // 0=default protocol (TCP)
  if (listenfd < 0) {
    unable_to_listen_error(args->port);
  }
  // Allow address (port number) to be reused immediately
  int optVal = 1;
  if (setsockopt(listenfd, SOL_SOCKET, SO_REUSEADDR, &optVal, sizeof(int)) <
      0) {
    unable_to_listen_error(args->port);
  }
  // Bind socket to address
  if (bind(listenfd, ai->ai_addr, sizeof(struct sockaddr)) < 0) {
    unable_to_listen_error(args->port);
  }
  // Indicate willingness to listen on socket - connections can now be queued
  if (listen(listenfd, args->clientLimit) <
      0) {  // Up to 10 connection requests can queue
    // (Reality on moss is that this queue length parameter is ignored)
    unable_to_listen_error(args->port);
  }
  // get number of port that server is listening on
  struct sockaddr_in bound;
  socklen_t len = sizeof(bound);
  if (getsockname(listenfd, (struct sockaddr *)&bound, &len) < 0) {
    unable_to_listen_error(args->port);
  }
  // print port that server is listenig on to terminal
  fprintf(stderr, "%hu\n", ntohs(bound.sin_port));
  // free ai
  freeaddrinfo(ai);
  // return listening socket
  return listenfd;
}

// Sends the predefined ‘invalid message’ response file, then terminates thread
// Inputs:  fd – connected client socket
// Returns: void (closes fd and calls pthread_exit)
void send_bad_prefix_error(int fd) {
  int responseFd =
      open(responseFile, O_CREAT | O_RDONLY, READ_WRITE_OWNER_PERMISSION);

  struct stat responseStat;
  if (fstat(responseFd, &responseStat) < 0) {
    return;
  }

  sendfile(fd, responseFd, NULL, responseStat.st_size);

  close(fd);
  close(responseFd);
  pthread_exit(NULL);
}

// Sends a protocol-level error message to a client and terminates thread
// Inputs:  fd    – connected client socket
//          error – NUL-terminated error string to send
// Returns: void (closes fd and calls pthread_exit)
void send_error_message(int fd, const char *const error) {
  // make prefix little endian and send
  uint32_t prefixLe = htole32(PREFIX);
  write(fd, &prefixLe, sizeof prefixLe);

  // send operation type error message
  uint8_t operationType = OPERATION_ERROR_MESSAGE;
  write(fd, &operationType, sizeof operationType);

  // obtain and send message length
  uint32_t messageSize = strlen(error);
  uint32_t messageSizeLe = htole32(messageSize);
  write(fd, &messageSizeLe, sizeof messageSizeLe);

  // send error message to client, close fd, exit from thread
  write(fd, error, messageSize);
  fflush(stdout);
  fsync(fd);
  close(fd);
  pthread_exit(NULL);
}

// Reads exactly n bytes from a file descriptor
// Inputs:  fd  – file descriptor to read from
//          buf – destination buffer
//          n   – number of bytes required
// Returns: 0 on success, –1 on EOF or error
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

// Reads a little-endian 32-bit unsigned integer from fd
// Inputs:  fd  – file descriptor
//          out – pointer receiving host-order value
// Returns: 0 on success, –1 on short read / error
int read_u32_le(int fd, uint32_t *out) {
  uint32_t tempLe;

  // read little endian value into a temporary variable
  if (read_exact(fd, &tempLe, sizeof tempLe) < 0) {
    return -1;
  }

  // change it from little ending back to "host byte order"
  *out = le32toh(tempLe);

  return 0;
}

// Streams imageSize bytes from socket fd into a temporary image file
// Inputs:  fd           – connected client socket
//          tempImageFd  – open, writable temp-file descriptor
//          imageSize    – number of bytes to transfer
// Returns: void (writes exactly imageSize bytes, or exits on error)
void copy_image_to_file(int fd, int tempImageFd, uint32_t imageSize) {
  // buffer to hold chunk of data
  uint8_t buffer[FOUR_KB_CHUNK];

  uint32_t remainingBytes = imageSize;

  while (remainingBytes > 0) {
    size_t bytesToRead = 0;

    // setup to read 4kb of data or whatevers left if less
    if (remainingBytes < sizeof buffer) {
      bytesToRead = remainingBytes;
    } else {
      bytesToRead = sizeof buffer;
    }

    // read from socket fd, then write to temp image fd
    ssize_t bytesRead;
    bytesRead = read(fd, buffer, bytesToRead);
    if (bytesRead <= 0) {
      return;
    }
    write(tempImageFd, buffer, bytesRead);

    // minus what we just read from the total size of the image we have to read
    if (remainingBytes > 0) {
      remainingBytes -= (uint32_t)bytesRead;
    }
  }
}

// Loads an incoming JPEG/PNG into an IplImage residing in memory
// Inputs:  fd        – connected client socket
//          imageSize – byte length obtained from protocol
//          mode      – loadDetectImage | loadReplaceImage string
// Returns: pointer to loaded IplImage on success, NULL on failure
//          (handles locking of shared temp image file)
IplImage *load_image_to_memory(int fd, uint32_t imageSize, const char *mode,
                               struct ThreadArgs *tArgs) {
  // open and lock the shared pathname using O_CLOEXEC (close on exec)
  pthread_mutex_lock(tArgs->lock);
  int tempImageFd =
      open(tempImage, O_CREAT | O_RDWR | O_TRUNC, READ_WRITE_OWNER_PERMISSION);
  if (tempImageFd < 0) {
    pthread_mutex_unlock(tArgs->lock);
    return NULL;
  }

  // send image from fd to tempImageFd
  copy_image_to_file(fd, tempImageFd, imageSize);

  IplImage *img;

  // load image to memory using cvLoadImage, if it errors, exit and send error
  if (strcmp(mode, loadDetectImage) == 0) {
    img = cvLoadImage(tempImage, CV_LOAD_IMAGE_COLOR);
  } else {
    img = cvLoadImage(tempImage, CV_LOAD_IMAGE_UNCHANGED);
  }

  // release lock, close fd, return the image
  close(tempImageFd);
  pthread_mutex_unlock(tArgs->lock);
  return img;
}

// Sends an already-saved output image back to the client
// Inputs:  fd           – connected client socket
//          outputFileFd – open file containing image to send
//          operation    – OPERATION_OUTPUT_IMAGE constant
// Returns: void (streams image and leaves fd open)
void send_image(int fd, int outputFileFd, int operation) {
  // make prefix little endian and send to client
  uint32_t prefixLe = htole32(PREFIX);
  write(fd, &prefixLe, sizeof prefixLe);

  // send operation to client
  uint8_t operationType = (uint8_t)operation;
  write(fd, &operationType, sizeof operationType);

  // create and populate struct with output file details, including size
  struct stat imageStat;
  if (fstat(outputFileFd, &imageStat) < 0) {
    return;
  }

  // convert size to little endian and send to client
  uint32_t imageSizeLe = htole32((uint32_t)imageStat.st_size);
  write(fd, &imageSizeLe, sizeof imageSizeLe);

  // send the output image to the client
  // offset means to start at the beginning of the outputFileFd
  off_t offset = 0;
  ssize_t totalSent = 0;
  while ((uint32_t)totalSent < imageStat.st_size) {
    ssize_t sent =
        sendfile(fd, outputFileFd, &offset, imageStat.st_size - totalSent);
    if (sent <= 0) {
      return;
    }

    // add what sendfile just sent to the total amount of bytes it has sent
    totalSent += sent;
  }
}

// Initialises face-replacement detection; sends error if no faces found
// Inputs:  fd          – client socket
//          frame       – image to modify (BGR)
//          replace     – replacement face image (BGR/BGRA)
//          faceCascade – out param for loaded cascade
//          frameGray   – out param for grayscale working image
//          storage     – out param for OpenCV storage
//          faces       – out param for detected face sequence
// Returns: 0 on success (may exit thread on error)
int init_face_replace(int fd, IplImage *frame, IplImage *replace,
                      CvHaarClassifierCascade **faceCascade,
                      IplImage **frameGray, CvMemStorage **storage,
                      CvSeq **faces) {
  // Load the face cascade
  *faceCascade =
      (CvHaarClassifierCascade *)cvLoad(faceCascadeFilename, NULL, NULL, NULL);

  // Convert to grayscale and equalize histogram
  *frameGray = cvCreateImage(cvGetSize(frame), IPL_DEPTH_8U, 1);
  cvCvtColor(frame, *frameGray, CV_BGR2GRAY);
  cvEqualizeHist(*frameGray, *frameGray);

  // Create and clear memory storage
  *storage = cvCreateMemStorage(0);
  cvClearMemStorage(*storage);

  // Detect faces in grayscale image
  *faces = cvHaarDetectObjects(*frameGray, *faceCascade, *storage,
                               haarScaleFactor, haarMinNeighbours, haarFlags,
                               cvSize(haarMinSize, haarMinSize),
                               cvSize(haarMaxSize, haarMaxSize));

  // if no faces found, send error message
  if (*faces == NULL || (*faces)->total == 0) {
    cvReleaseImage(&frame);
    cvReleaseImage(&replace);
    cvReleaseImage(frameGray);
    cvReleaseHaarClassifierCascade(faceCascade);
    cvReleaseMemStorage(storage);
    send_error_message(fd, noFacesDetectedError);
  }
  return 0;
}

// Overlays each detected face in ‘frame’ with the resized ‘replace’ sprite
// Inputs:  frame   – target image whose faces will be overwritten
//          replace – face sprite (BGR or BGRA with alpha channel)
//          faces   – sequence of CvRect face regions
// Returns: void (modifies frame in-place)
void process_face_replace(IplImage *frame, IplImage *replace, CvSeq *faces) {
  // Iterate through detected faces and replace them
  for (int i = 0; i < faces->total; i++) {
    CvRect *face = (CvRect *)cvGetSeqElem(faces, i);

    IplImage *resized = cvCreateImage(cvSize(face->width, face->height),
                                      IPL_DEPTH_8U, replace->nChannels);

    // Resize the replacement image
    cvResize(replace, resized, CV_INTER_AREA);

    char *frameData = frame->imageData;
    char *faceData = resized->imageData;

    // Copy each pixel from the resized replacement image
    for (int y = 0; y < face->height; y++) {
      for (int x = 0; x < face->width; x++) {
        int faceIndex = (resized->widthStep * y) + (x * resized->nChannels);

        // Skip pixel if BGRA and alpha channel is 0
        if (resized->nChannels == bgraChannels &&
            faceData[faceIndex + alphaIndex] == 0) {
          continue;
        }

        int frameIndex = (frame->widthStep * (face->y + y)) +
                         ((face->x + x) * frame->nChannels);

        // Overwrite BGR channels
        frameData[frameIndex + 0] = faceData[faceIndex + 0];
        frameData[frameIndex + 1] = faceData[faceIndex + 1];
        frameData[frameIndex + 2] = faceData[faceIndex + 2];
      }
    }

    cvReleaseImage(&resized);
  }
}

// Saves the modified frame, streams it to client, and frees OpenCV objects
// Inputs:  fd           – client socket
//          frame        – modified colour image
//          replace      – replacement sprite (already used)
//          frameGray    – grayscale working copy
//          faceCascade  – loaded Haar cascade
//          storage      – OpenCV storage to release
// Returns: 0 on success, –1 on I/O error before send
int save_and_send_replacement(int fd, IplImage *frame, IplImage *replace,
                              IplImage *frameGray,
                              CvHaarClassifierCascade *faceCascade,
                              CvMemStorage *storage, struct ThreadArgs *tArgs) {
  // open and lock the shared pathname using
  pthread_mutex_lock(tArgs->lock);
  int outputFileFd =
      open(tempImage, O_CREAT | O_RDWR | O_TRUNC, READ_WRITE_OWNER_PERMISSION);
  if (outputFileFd < 0) {
    pthread_mutex_unlock(tArgs->lock);
    return -1;
  };

  // Save the output image
  cvSaveImage(tempImage, frame, 0);
  fsync(outputFileFd);

  // send modified file back to client
  send_image(fd, outputFileFd, OPERATION_OUTPUT_IMAGE);

  // close fd
  fflush(stdout);
  fsync(outputFileFd);
  close(outputFileFd);

  // Cleanup and release lock
  cvReleaseImage(&frame);
  cvReleaseImage(&replace);
  cvReleaseImage(&frameGray);
  cvReleaseHaarClassifierCascade(&faceCascade);
  cvReleaseMemStorage(&storage);

  pthread_mutex_unlock(tArgs->lock);

  return 0;
}

// High-level wrapper: perform face replacement workflow on a single request
// Inputs:  fd      – client socket
//          frame   – image to search
//          replace – face sprite
// Returns: 0 on success (errors are sent inside helpers)
int face_replace(int fd, IplImage *frame, IplImage *replace,
                 struct ThreadArgs *tArgs) {
  // setup variables
  CvHaarClassifierCascade *faceCascade;
  IplImage *frameGray;
  CvMemStorage *storage;
  CvSeq *faces;

  // initialise and populate variables
  init_face_replace(fd, frame, replace, &faceCascade, &frameGray, &storage,
                    &faces);

  // replace faces in frame with replace
  process_face_replace(frame, replace, faces);

  // send modified image to client
  save_and_send_replacement(fd, frame, replace, frameGray, faceCascade, storage,
                            tArgs);

  return 0;
}

// Initialises face & eye cascades and detects faces; exits if none found
// Inputs:  fd            – client socket
//          frame         – colour image from client
//          faceCascade   – out param for face cascade
//          eyesCascade   – out param for eye cascade
//          frameGray     – out param grayscale buffer
//          storage       – out param OpenCV storage
//          faces         – out param face sequence
// Returns: 0 on success (otherwise sends error & exits thread)
int init_face_detection(int fd, IplImage *frame,
                        CvHaarClassifierCascade **faceCascade,
                        CvHaarClassifierCascade **eyesCascade,
                        IplImage **frameGray, CvMemStorage **storage,
                        CvSeq **faces) {
  // Load Haar cascades
  *faceCascade =
      (CvHaarClassifierCascade *)cvLoad(faceCascadeFilename, NULL, NULL, NULL);
  *eyesCascade =
      (CvHaarClassifierCascade *)cvLoad(eyesCascadeFilename, NULL, NULL, NULL);
  if (!*faceCascade || !*eyesCascade) return -1;

  // Convert to grayscale + histogram equalize
  *frameGray = cvCreateImage(cvGetSize(frame), IPL_DEPTH_8U, 1);
  cvCvtColor(frame, *frameGray, CV_BGR2GRAY);
  cvEqualizeHist(*frameGray, *frameGray);

  // Prepare storage + detect faces
  *storage = cvCreateMemStorage(0);
  cvClearMemStorage(*storage);
  *faces = cvHaarDetectObjects(*frameGray, *faceCascade, *storage,
                               haarScaleFactor, haarMinNeighbours, haarFlags,
                               cvSize(haarMinSize, haarMinSize),
                               cvSize(haarMaxSize, haarMaxSize));

  // cleanup and send error if no faces are found
  if (*faces == NULL || (*faces)->total == 0) {
    cvReleaseImage(&frame);
    cvReleaseImage(frameGray);
    cvReleaseHaarClassifierCascade(faceCascade);
    cvReleaseHaarClassifierCascade(eyesCascade);
    cvReleaseMemStorage(storage);
    send_error_message(fd, noFacesDetectedError);
  }
  return 0;
}

// Draws magenta ellipses over faces and blue circles over exactly two eyes
// Inputs:  frame       – source image (will be annotated)
//          frameGray   – grayscale copy used for eye detection
//          eyesCascade – pre-loaded eye cascade
//          faces       – sequence of detected faces
// Returns: void (annotations drawn directly onto frame)
void process_detected_faces(IplImage *frame, IplImage *frameGray,
                            CvHaarClassifierCascade *eyesCascade,
                            CvSeq *faces) {
  for (int i = 0; i < faces->total; i++) {
    CvRect *face = (CvRect *)cvGetSeqElem(faces, i);
    CvPoint center = {face->x + face->width / 2, face->y + face->height / 2};
    const CvScalar magenta = cvScalar(255, 0, 255, 0);
    cvEllipse(frame, center, cvSize(face->width / 2, face->height / 2), 0,
              ellipseStartAngle, ellipseEndAngle, magenta, lineThickness,
              lineType, shift);

    // Isolate face ROI for eye detection
    IplImage *faceROI = cvCreateImage(cvGetSize(frameGray), IPL_DEPTH_8U, 1);
    cvCopy(frameGray, faceROI, NULL);
    cvSetImageROI(faceROI, *face);

    CvMemStorage *eyeStorage = cvCreateMemStorage(0);
    cvClearMemStorage(eyeStorage);
    CvSeq *eyes = cvHaarDetectObjects(
        faceROI, eyesCascade, eyeStorage, haarScaleFactor, haarMinNeighbours,
        haarFlags, cvSize(haarMinSize, haarMinSize),
        cvSize(haarMaxSize, haarMaxSize));

    if (eyes->total == 2) {
      const CvScalar blue = cvScalar(255, 0, 0, 0);
      for (int j = 0; j < eyes->total; j++) {
        CvRect *eye = (CvRect *)cvGetSeqElem(eyes, j);
        CvPoint eyeCenter = {face->x + eye->x + eye->width / 2,
                             face->y + eye->y + eye->height / 2};
        int radius = cvRound((eye->width / 2 + eye->height / 2) / 2);
        cvCircle(frame, eyeCenter, radius, blue, lineThickness, lineType,
                 shift);
      }
    }
    cvReleaseImage(&faceROI);
    cvReleaseMemStorage(&eyeStorage);
  }
}

// Saves annotated image, streams it to client, cleans up all resources
// Inputs:  fd           – client socket
//          frame        – annotated image
//          frameGray    – grayscale buffer
//          faceCascade  – loaded face cascade
//          eyesCascade  – loaded eye cascade
//          storage      – OpenCV storage to release
// Returns: 0 on success, –1 on I/O error before send
int save_and_send_image(int fd, IplImage *frame, IplImage *frameGray,
                        CvHaarClassifierCascade *faceCascade,
                        CvHaarClassifierCascade *eyesCascade,
                        CvMemStorage *storage, struct ThreadArgs *tArgs) {
  // lock and open file
  pthread_mutex_lock(tArgs->lock);
  int outputFileFd =
      open(tempImage, O_CREAT | O_RDWR | O_TRUNC, READ_WRITE_OWNER_PERMISSION);
  if (outputFileFd < 0) {
    pthread_mutex_unlock(tArgs->lock);
    return -1;
  }

  // save image to temp file and then send to client, then close fd
  cvSaveImage(tempImage, frame, 0);
  fsync(outputFileFd);
  send_image(fd, outputFileFd, OPERATION_OUTPUT_IMAGE);
  close(outputFileFd);

  // Cleanup all OpenCV objects and release lock
  cvReleaseImage(&frame);
  cvReleaseImage(&frameGray);
  cvReleaseHaarClassifierCascade(&faceCascade);
  cvReleaseHaarClassifierCascade(&eyesCascade);
  cvReleaseMemStorage(&storage);
  pthread_mutex_unlock(tArgs->lock);
  return 0;
}

// Full face-detection request handler (detect + annotate)
// Inputs:  fd    – client socket
//          frame – image to analyse
// Returns: 0 on success (any protocol errors handled internally)
int face_detect(int fd, IplImage *frame, struct ThreadArgs *tArgs) {
  // setup variables
  CvHaarClassifierCascade *faceCascade, *eyesCascade;
  IplImage *frameGray;
  CvMemStorage *storage;
  CvSeq *faces;

  // populate variables
  init_face_detection(fd, frame, &faceCascade, &eyesCascade, &frameGray,
                      &storage, &faces);

  // detect faces abd eyes in image
  process_detected_faces(frame, frameGray, eyesCascade, faces);

  // return the adjusted image to client
  save_and_send_image(fd, frame, frameGray, faceCascade, eyesCascade, storage,
                      tArgs);

  return 0;
}

// Retrieves and validates the next uint32_t image size from client
// Inputs:  fd – client socket
// Returns: validated image size (may send error and never return on failure)
uint32_t get_image_size(int fd, struct ThreadArgs *tArgs) {
  uint32_t imageSize = 0;
  if (read_u32_le(fd, &imageSize) < 0) {
    // TODO: probably wrong error message?
    send_error_message(fd, imageLoadFailureError);
  } else if (imageSize == 0) {
    send_error_message(fd, zeroByteImageError);
  } else if (imageSize > tArgs->maxSize) {
    send_error_message(fd, imageTooLargeError);
  }
  return imageSize;
}

// Thread routine: handles one client request (detect or replace)
// Inputs:  arg – pointer to malloc’d ThreadArgs struct with thread context
// Returns: NULL (thread exits on completion; fd closed)
void *client_thread(void *arg) {
  struct ThreadArgs *tArgs = (struct ThreadArgs *)arg;
  int fd = tArgs->fd;

  // read prefix, send error if incorrect
  uint32_t prefix = 0;
  if (read_u32_le(fd, &prefix) < 0 || prefix != PREFIX) {
    send_bad_prefix_error(fd);
  }

  // read operation type should be face detect or replace from client
  uint8_t operationType = 0;
  if (read_exact(fd, &operationType, 1) < 0) {
    send_error_message(fd, invalidProtocolError);
  } else if (operationType != OPERATION_FACE_DETECTION &&
             operationType != OPERATION_FACE_REPLACEMENT) {
    send_error_message(fd, invalidOperationError);
  }

  // get size of detect image, send error message if anything goes wrong
  uint32_t detectImageSize = get_image_size(fd, tArgs);

  // load detect image into memory
  IplImage *detectImage =
      load_image_to_memory(fd, detectImageSize, loadDetectImage, tArgs);
  if (!detectImage) {
    send_error_message(fd, imageLoadFailureError);
  }

  // if operation is fact detect
  if (operationType == OPERATION_FACE_DETECTION) {
    // perform face detection and send image to client
    face_detect(fd, detectImage, tArgs);

    // if operation is face replace
  } else if (operationType == OPERATION_FACE_REPLACEMENT) {
    // get size of replace image, send error message if anything goes wrong
    uint32_t replaceImageSize = get_image_size(fd, tArgs);

    IplImage *replaceImage =
        load_image_to_memory(fd, replaceImageSize, loadReplaceImage, tArgs);

    // run face replace and send to client
    face_replace(fd, detectImage, replaceImage, tArgs);
  }

  close(tArgs->fd);
  free(tArgs);
  return NULL;
}

// Accepts client connections up to clientLimit and dispatches threads
// Inputs:  fdServer – listening socket
//          args     – pointer to immutable Args (clientLimit, etc.)
// Returns: void (runs forever unless accept fails)
void process_connections(int fdServer, const struct Args *args,
                         pthread_mutex_t *lock) {
  int fd;
  struct sockaddr_in fromAddr;
  socklen_t fromAddrSize;

  // Repeatedly accept connections and process data
  while (1) {
    fromAddrSize = sizeof(struct sockaddr_in);
    // Block, waiting for a new connection.
    fd = accept(fdServer, (struct sockaddr *)&fromAddr, &fromAddrSize);
    if (fd < 0) {
      unable_to_listen_error(args->port);
    }

    // create a thread to deal with client, create shared context with mutex
    struct ThreadArgs *tArgs = malloc(sizeof *tArgs);
    tArgs->fd = fd;
    tArgs->maxSize = args->maxSize;
    tArgs->lock = lock;

    pthread_t threadID;
    // pthread_create(thread_id, attributes, routine, sole arg of routine)
    // make a thread which executes client_thread with arg of data
    pthread_create(&threadID, NULL, client_thread, tArgs);
    pthread_detach(threadID);
  }
}

// Program entry point: validates CLI, sets up server, enters accept loop
// Inputs:  argc / argv – raw command-line arguments
// Returns: EXIT_SUCCESS on normal termination (though loop is infinite)
int main(int argc, char *argv[]) {
  // Validate command line arguments
  if (invalid_command_line_args(argc, argv)) {
    fprintf(stderr, "%s", usageError);
    exit(USAGE_ERROR_EXIT);
  }

  // Populate struct with command line arguments
  struct Args *args = calloc(1, sizeof(struct Args));
  populate_args_struct(argc, argv, args);

  // check temp image has write permissions
  temporary_image_file_check();

  // check that cascase classifiers can be loaded
  cascade_classifier_file_check();

  // listens on a given port, prints the port it's listening to to stdout
  int fdServer = open_listen(args);

  // begin utilising threads with shared mutex to handle client requests
  pthread_mutex_t lock;
  pthread_mutex_init(&lock, NULL);
  process_connections(fdServer, args, &lock);

  close(fdServer);
  pthread_mutex_destroy(&lock);
  free_args(args);

  return 0;
}