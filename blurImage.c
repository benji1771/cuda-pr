#include <SDL2/SDL.h>
#include <SDL2/SDL_image.h>
#include <stdio.h>
#include <time.h>
#include <stdbool.h>

// code from class
// Cleans up and exits
void cleanupAndClose(int exitCode);

// Load a given image
SDL_Surface* loadImage(char *filename);

// blurImage
void blurImage(SDL_Surface *image, int a);


void checkSDL(void* result);
SDL_Surface *source;

int main(int argc, char *argv[])
{
    // Check that we have the right number of args.
    if(argc != 4)
    {
        fprintf(stderr, "Usage: %s source \n", argv[0]);
        cleanupAndClose(EXIT_FAILURE);
    }

    // We not do need to initialize SDL to load image data so we get right to
    // loading the source image.
    int a = atoi(argv[1]);

    checkSDL(source = loadImage(argv[2]));

    printf("grey..ing?\n");
    blurImage(source, a);

	// Save image
    printf("Saving: %s\n", argv[3]);
    IMG_SavePNG(source, argv[3]);
    // I have heard that the PNG writer is buggy... I have not had any trouble, but you
    // can also use the BMP writer (hard to mess up BMPs).
    //SDL_SaveBMP(source, "shiftOut.bmp");

    cleanupAndClose(EXIT_SUCCESS);
    return EXIT_SUCCESS;
}
void checkSDL(void* result)
{
    if(result == NULL)
    {
        fprintf(stderr, "Error: %s\n", SDL_GetError());
        cleanupAndClose(EXIT_FAILURE);
    }
}
void cleanupAndClose(int exitCode)
{
    if(source) SDL_FreeSurface(source);
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


void blurImage(SDL_Surface *image, int a)
{
	Uint32 *pixels = (Uint32 *)image->pixels;
    for(int y = 0; y < image->h; y++)
    {
        for(int x = 0; x < image->w; x++)
        {
            Uint32 pixel = pixels[(y * image->w) + x];
        	Uint32 a = (pixel & image->format->Amask);


            int xbegin = x - a;
            int ybegin = y - a;
            int xend = x + a;
            int yend = x + a;
            Uint32 avg = pixel;
            int count = 1;
            for(int piy = ybegin; piy < yend; piy++){
                if(piy < 0 || piy >= image->h) continue;
                for(int pix = xbegin; pix < xend; pix++){
                    if(pix < 0 || pix >= image->w || (pix == x && piy == y)) continue;
                    avg += pixels[(piy * image->w) + pix];
                    count++;
                }
            }
            
            pixel = avg / count;
            Uint32 r = (pixel & image->format->Rmask); // Isolate red component


            Uint32 g = (pixel & image->format->Gmask); // Isolate green component


            Uint32 b = (pixel & image->format->Bmask); // Isolate blue component


            
		
                // Build the grey pixels
            pixels[(y * image->w) + x] = (r) | 
                                         (g) | 
                                         (b) | 
                                         (a);
        }
    }
}
