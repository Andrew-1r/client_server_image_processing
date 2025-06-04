/*
 * csse2310a4.h
 */

#ifndef CSSE2310A4_H
#define CSSE2310A4_H

#include <stdio.h>
#include <stdbool.h>

char** split_by_char(char* str, char split, unsigned int maxFields);

typedef struct {
    char* name;
    char* value;
} HttpHeader;

void free_header(HttpHeader* header);
void free_array_of_headers(HttpHeader** headers);

int get_HTTP_request(FILE* f, char** method, char** address, 
        HttpHeader*** headers, unsigned char** body, unsigned long* bodySize);

unsigned char* construct_HTTP_response(int status, const char* statusExplanation, 
	HttpHeader** headers, const unsigned char* body, 
        unsigned long bodySize, unsigned long* len);

int get_HTTP_response(FILE* f, int* httpStatus, char** statusExplain, 
        HttpHeader*** headers, unsigned char** body, unsigned long* bodySize);

#endif
