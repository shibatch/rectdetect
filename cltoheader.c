// Copyright Naoki Shibata 2018. Distributed under the MIT License.

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>

void escesc(char *out, char *in, int outlen) {
  int ip = 0, op = 0, state = 0;

  while(op < outlen-1) {
    switch(state) {
      case 0:
	if (in[ip] == '\"') {
	  ip++;
	  out[op++] = '\\';
	  if (op >= outlen-1) break;
	  out[op++] = '\"';
	  if (op >= outlen-1) break;
	  state = 1;
	} else {
	  out[op++] = in[ip++];
	}
	break;
      case 1:
	if (in[ip] == '\"') {
	  ip++;
	  out[op++] = '\\';
	  if (op >= outlen-1) break;
	  out[op++] = '\"';
	  if (op >= outlen-1) break;
	  state = 0;
	} else if (in[ip] == '\\') {
	  ip++;
	  out[op++] = '\\';
	  if (op >= outlen-1) break;
	  out[op++] = '\\';
	  if (op >= outlen-1) break;
	} else {
	  out[op++] = in[ip++];
	}
	break;
    }
  }

  out[op] = '\0';
}

int main(int argc, char **argv) {
  if (argc < 3) {
    fprintf(stderr, "Usage : %s <input cl file> <varname>\n", argv[0]);
    exit(-1);
  }

  FILE *fp = fopen(argv[1], "r");
  if (fp == NULL) {
    fprintf(stderr, "Counldn't open %s\n", argv[1]);
    exit(-1);
  }

  printf("static const char *%s =\n", argv[2]);

  char line[1024], line2[1024];

  for(;;) {
    if (fgets(line, 1000, fp) == NULL) break;;
    for(int i=0;i<1024;i++) {
      if (line[i] == '\0') break;
      if (line[i] == '\n') {
	line[i] = '\0';
	break;
      }
    }
    escesc(line2, line, 1000);
    printf("\"%s\\n\"\n", line2);
  }

  printf(";\n");

  fclose(fp);
  exit(0);
}

