#include <SDL2/SDL.h>
#include <SDL2/SDL_image.h>
#include <stdio.h>
#include <time.h>
#include <stdbool.h>

// Cleans up and exits
void cleanupAndClose(int exitCode);

// Load a given image
SDL_Surface* loadImage(char *filename);

// grey image out
__global__ void greyImage(Uint32 *pixels, int w, int h);

SDL_Surface *source = NULL;
Uint32 *pixels = NULL;

int main(int argc, char *argv[])
{
    // Check that we have the right number of args.
    if(argc != 3)
    {
        fprintf(stderr, "Usage: %s source \n", argv[0]);
        cleanupAndClose(EXIT_FAILURE);
    }

    // We not do need to initialize SDL to load image data so we get right to
    // loading the source image.
    if((source = loadImage(argv[1])) == NULL)
    {
        fprintf(stderr, "Error: %s\n", SDL_GetError());
        cleanupAndClose(EXIT_FAILURE);
    }
    int N = source->h * source->w;
    int THREADS = 256;
    int BLOCKS = (N + THREADS - 1 ) / THREADS;

    // Copy the pixels to the GPU (add error checking)
    printf("Copying pixels to GPU\n");
    cudaMalloc(&pixels, sizeof(Uint32) * source->h * source->w);
    cudaMemcpy(pixels, source->pixels, sizeof(Uint32) * source->h * source->w, cudaMemcpyHostToDevice);

    printf("cuda grey...ing?\n");
    greyImage<<<BLOCKS,THREADS>>>(pixels, source->w, source->h);

    // Copy the pixels back to the host (add error checking)
    printf("Copying pixels to CPU\n");
    cudaMemcpy(source->pixels, pixels, sizeof(Uint32) * source->h * source->w, cudaMemcpyDeviceToHost);

	// Save image
    printf("Saving: %s\n", argv[2]);
    IMG_SavePNG(source, argv[2]);

    cleanupAndClose(EXIT_SUCCESS);
    return EXIT_SUCCESS;
}

void cleanupAndClose(int exitCode)
{
    if(source) SDL_FreeSurface(source);
    if(pixels) cudaFree(pixels);
    exit(exitCode);
}

SDL_Surface* loadImage(char *filename)
{
    printf("Loading: %s\n", filename);

    // Load the image into memory
    SDL_Surface *image = IMG_Load(filename);

    // Make sure pixel values are 32-bit
    if(image)
    {
    	SDL_Surface *temp = SDL_ConvertSurfaceFormat(image, SDL_PIXELFORMAT_RGBA32, 0);
    	SDL_FreeSurface(image);
    	image = temp;
    }

    return image;
}

// A still less than optimal shifter...
__global__ void greyImage(Uint32 *pixels, int w, int h)
{
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    if (y >= h || x >= w) return;

    Uint32 pixel = pixels[(y * w) + x];
    Uint32 r = pixel & 0x000000ff; // Isolate red component
    Uint32 g = pixel & 0x0000ff00; // Isolate green component
    g = g >> 8;                    // Shift it down
    Uint32 b = pixel & 0x00ff0000; // Isolate blue component
    b = b >> 16;                   // Shift it down
    Uint32 a = pixel & 0xff000000; // Isolate alpha component
    a = a >> 24;                   // Shift it down

    Uint32 newPix = 0.21f * r + 0.71f * g + 0.07f * b;
    // Build the shifted pixel
    pixels[(y * w) + x] = newPix | (newPix << 8) | (newPix << 16) | (a << 24) ;
        
    
}
