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

// shift red -> green -> blue -> red in the given image
void greyimage(SDL_Surface *image);


void checkSDL(void* result);
SDL_Surface *source;

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
    checkSDL(source = loadImage(argv[1]));

    printf("grey..ing?\n");
    greyimage(source);

	// Save image
    printf("Saving: %s\n", argv[2]);
    IMG_SavePNG(source, argv[2]);
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


void greyimage(SDL_Surface *image)
{
	Uint32 *pixels = (Uint32 *)image->pixels;
    for(int y = 0; y < image->h; y++)
    {
        for(int x = 0; x < image->w; x++)
        {
        	Uint32 pixel = pixels[(y * image->w) + x];

        	Uint32 r = (pixel & image->format->Rmask); // Isolate red component
		r = r >> image->format->Rshift;
		r = r * 0.21f;
        	Uint32 g = (pixel & image->format->Gmask); // Isolate green component
		g = g >> image->format->Gshift;
		g = g * 0.71f;
        	Uint32 b = (pixel & image->format->Bmask); // Isolate blue component
		b = b >> image->format->Bshift;
		b = b * 0.07f;
	        Uint32 a = pixel & image->format->Amask;
		a = a >> image->format->Ashift;
			// Build the grey pixels
		pixels[(y * image->w) + x] = (r << image->format->Rshift) |
					     (g << image->format->Gshift) |
   					     (b << image->format->Bshift) |
					     (a << image->format->Ashift);
        }
    }
}

