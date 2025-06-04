Two seperate programs; a client and a server that uses multithreading. 

The server gives it's port number when run, a large amount of clients can then connect to it and send images to have faces/eyes detected or replaced.

# Usage

Unzip the project into a folder in linux.

`cd <folder-you-installed-to>`

`make`

`./uqfacedetectserver 0 0`
- in one terminal, will open up server and print port to stderr

`./uqfacedetect --usage`
- shows you how to send requests to the server

The task sheet contains hints on usage if you get stuck 😊.