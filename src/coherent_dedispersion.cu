#include <cuda_runtime.h>
#include <cuComplex.h>
#include "../src/cudaErrorChecking.h"

#include <GL/glew.h>
#include <GLFW/glfw3.h>

#include <stdlib.h>
#include <stdio.h>

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

#include <cuda_gl_interop.h>

// GLUT-related constants
#define OPEN_FILE  1

// Mouse states
static double xprev;
static double yprev;
static bool drag_mode;

// Window states
static float windowWidth;
static float windowHeight;

#define XNORM(xpos)  ( (xpos)/windowWidth - 0.5)
#define YNORM(ypos)  (-(ypos)/windowHeight + 0.5)

static struct cudaGraphicsResource *cudaPointsResource;
float *d_points;

static struct cudaGraphicsResource *cudaImageResource;
texture<float, 2, cudaReadModeElementType> d_image;
cudaSurfaceObject_t surf;
cudaResourceDesc surfRes;
cudaArray *cuArray;
cudaChannelFormatDesc channelDesc;

int w, h;

/**
 * Convert a VDIF buffer into an array of floats
 *
 * @param in              Pointer to the VDIF buffer
 * @param out             Pointer to the output buffer
 * @param frameSizeBytes  The number of bytes per VDIF frame
 *                        (including the header)
 * @param headerSizeBytes The number of bytes per VDIF frame header
 */
__global__ void cudaVDIFToFloatComplex( char2 *in, cuFloatComplex *out, int frameSizeBytes, int headerSizeBytes )
{
    // The size of just the data part of the frame
    int dataSizeBytes = frameSizeBytes - headerSizeBytes;

    // It is assumed that in points to the first byte in a frameheader
    int i = threadIdx.x + blockIdx.x*blockDim.x; // Index of (non-header) data sample

    // Express the index in terms of bytes
    int i2 = i*sizeof(char2);

    // Get the frame number for this byte, and the idx within this frame
    int frame      = i2 / dataSizeBytes;
    int idxInFrame = i2 % dataSizeBytes;

    // Calculate the indices into the input and output arrays for this sample
    int in_idx  = frame*frameSizeBytes + (headerSizeBytes + idxInFrame);
    int out_idx = i;

    // Bring the sample to register memory
    char2 sample = in[in_idx];

    // Turn it into a float and write it to global memory
    out[out_idx] = make_cuFloatComplex( (float)sample.x - 128.0, sample.y - 128.0 );
}

/**
 * Apply a phase ramp to complex data
 *
 * @param data          The data to which the phase ramp is applied (in-place)
 * @param radPerBin     The slope of the phase ramp (in radians per bin)
 * @param samplesPerBin The number of contiguous samples to be rotated by the
 *                      same amount
 */
__global__ void cudaApplyPhaseRamp( cuFloatComplex *data, float radPerBin, int samplesPerBin )
{
    // For this block/thread...
    int s = threadIdx.x + blockIdx.x*blockDim.x; // Get the (s)ample number
    int b = s / samplesPerBin;                   // Get the (b)in number

    // For each bin, calculate the phase rotation to be applied
    float rad = b * radPerBin;
    cuFloatComplex phase;
    sincosf( rad, &phase.y, &phase.x );

    // Apply the phase ramp (in-place)
    data[s] = cuCmulf( data[s], phase );
}

/**
 * Convert dual polarisation data to Stokes I
 *
 * @param data The data to be converted
 * @param stokesI The Stokes I output
 *
 * `data` is expected to be an array of *pairs* of complex numbers,
 * X,Y,X,Y,X,Y,...
 * from which the Stokes parameters are formed:
 *    I = |X|^2 + |Y|^2
 */
__global__ void cudaStokesI( cuFloatComplex *data, float *stokesI )
{
    // Let i represent the output sample index
    int i = threadIdx.x + blockIdx.x*blockDim.x;

    // Pull out the two polarisations
    cuFloatComplex X = data[2*i];
    cuFloatComplex Y = data[2*i + 1];

    // Calculate Stokes I
    stokesI[i] = X.x*X.x + X.y*X.y + Y.x*Y.x + Y.y*Y.y;
}


/*
int glut_main( int argc, char **argv )
{
    // Use a single buffered window in RGB mode (as opposed to a double-buffered
    // window or color-index mode).
    glutInit( &argc, argv );
    glutInitDisplayMode( GLUT_SINGLE | GLUT_RGB );

    // Position window at (80,80)-(480,380) and give it a title.
    glutInitWindowPosition( 80, 80 );
    glutInitWindowSize( 400, 300 );
    glutCreateWindow( "A Simple Triangle" );

    // Tell GLUT that whenever the main window needs to be repainted that it
    // should call the function display().
    glutDisplayFunc( display );

    // Prepare some test data
    size_t nFrames            = 4;
    size_t frameSizeBytes     = 544;
    size_t headerSizeBytes    = 32;
    size_t dataSizeBytes      = frameSizeBytes - headerSizeBytes;
    size_t nSamples           = nFrames * dataSizeBytes / sizeof(char2);
    size_t nPols              = 2;
    size_t nDualPolSamples    = nSamples / nPols;
    size_t vdifSizeBytes      = frameSizeBytes*nFrames;
    size_t vdifDataSizeBytes = nSamples * sizeof(cuFloatComplex);
    size_t stokesISizeBytes   = nDualPolSamples * sizeof(float);

    // Allocate memory
    char2 *vdif, *d_vdif;
    cuFloatComplex *d_vdifData;
    float *d_StokesI;

    gpuErrchk( cudaMallocHost( &vdif, vdifSizeBytes ) );
    gpuErrchk( cudaMalloc( &d_vdif, vdifSizeBytes ) );
    gpuErrchk( cudaMalloc( &d_vdifData, vdifDataSizeBytes ) );
    gpuErrchk( cudaMalloc( &d_StokesI, stokesISizeBytes ) );

    FILE *f = fopen( "../tests/testdata.vdif", "r" );
    fread( vdif, vdifSizeBytes, 1, f );
    fclose( f );

    // Load it up and strip the headers
    gpuErrchk( cudaMemcpy( d_vdif, vdif, nFrames * frameSizeBytes, cudaMemcpyHostToDevice ) );
    gpuErrchk( cudaDeviceSynchronize() );

    cudaVDIFToFloatComplex<<<nSamples/1024, 1024>>>( d_vdif, d_vdifData, frameSizeBytes, headerSizeBytes );
    gpuErrchk( cudaDeviceSynchronize() );

    cudaStokesI<<<nDualPolSamples/1024, 1024>>>( d_vdifData, d_StokesI );
    gpuErrchk( cudaDeviceSynchronize() );

    // Tell GLUT to start reading and processing events.  This function
    // never returns; the program only exits when the user closes the main
    // window or kills the process.
    glutMainLoop();

    // The following is never reached!!
    // Clean up memory
    gpuErrchk( cudaFree( d_vdif ) );
    gpuErrchk( cudaFree( d_vdifData ) );
    gpuErrchk( cudaFree( d_StokesI ) );
    gpuErrchk( cudaFreeHost( vdif ) );
}
*/

void mouse_button_callback( GLFWwindow *window, int button, int action, int mods )
{
    if (button == GLFW_MOUSE_BUTTON_LEFT)
    {
        size_t size;
        switch (action)
        {
            case GLFW_PRESS:
                glfwGetCursorPos( window, &xprev, &yprev );
                drag_mode = true;
                gpuErrchk( cudaGraphicsMapResources( 1, &cudaPointsResource, 0 ) );
                gpuErrchk( cudaGraphicsResourceGetMappedPointer( (void **)&d_points, &size, cudaPointsResource ) );
                break;
            case GLFW_RELEASE:
                drag_mode = false;
                gpuErrchk( cudaGraphicsUnmapResources( 1, &cudaPointsResource, 0 ) );
                break;
        }
    }
}

__global__
void cudaCreateImage( cudaSurfaceObject_t surf, int width, int height )
{
    // Makes an image of pixels ranging from 0.0 to 1.0, arranged in a gradient
    // so that top left is 0.0, bottom right is 1.0
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    // Get normalised Manhattan distance
    float dist = (x + y)/(float)(width + height - 2);

    // Set the pixel value, with the peak being at the centre
    surf2Dwrite( dist, surf, x*sizeof(float), y );
}

__global__
void cudaRotatePoints( float *points, float rad )
{
    // Assumes "points" is an array of sets of (x,y) coords
    // (i.e. two floats per point), with stride 4
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    int stride = 4;
    float x = points[stride*i];
    float y = points[stride*i+1];
    float s, c;
    sincosf( rad, &s, &c );
    points[stride*i]   = c*x - s*y;
    points[stride*i+1] = s*x + c*y;
}

__global__
void cudaChangeBrightness( float *image, float amount )
{
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    image[i] += amount;
}

void cursor_position_callback( GLFWwindow* window, double xpos, double ypos )
{
    if (drag_mode)
    {
        float rad = atan2(YNORM(ypos),  XNORM(xpos)) -
                    atan2(YNORM(yprev), XNORM(xprev));
        //float dy = YNORM(ypos) - YNORM(yprev);

        // OpenGL CUDA interoperability
        cudaRotatePoints<<<1,4>>>( d_points, rad );
        //cudaChangeBrightness<<<1,36>>>( d_image, dy );

        xprev = xpos;
        yprev = ypos;
    }
}

// This function allocates memory
char *loadFileContentsAsStr( const char *filename )
{
    // Open the file for reading
    FILE *f = fopen( filename, "r" );
    if (f == NULL)
    {
        fprintf( stderr, "error: loadFileContentsAsStr: unable to open file "
                "%s\n", filename );
        exit(EXIT_FAILURE);
    }

    // Get the size of the file
    fseek( f, 0L, SEEK_END );
    long size = ftell( f );
    rewind( f );

    // Allocate memory in a string buffer
    char *str = (char *)malloc( size + 1 );

    // Read in the file contents to the string buffer
    long nread = fread( str, 1, size, f );
    if (nread != size)
    {
        fprintf( stderr, "warning: loadFileContentsAsStr: reading in "
                "contents of %s truncated (%ld/%ld bytes read)\n",
                filename, nread, size );
    }

    // Put a null termination at the end
    str[size] = '\0';

    return str;
}

int main( int argc, char *argv[] )
{
    // Start GL context and O/S window using the GLFW helper library
    glfwInit();
    const char *glfwerr;
    int code = glfwGetError( &glfwerr );
    if (code != GLFW_NO_ERROR)
    {
        fprintf( stderr, "ERROR: could not start GLFW3: %s\n", glfwerr );
        return EXIT_FAILURE;
    }

    windowWidth = 640;
    windowHeight = 480;
    GLFWwindow* window = glfwCreateWindow( windowWidth, windowHeight, "DM Slider", NULL, NULL );
    if (!window)
    {
        fprintf(stderr, "ERROR: could not open window with GLFW3\n");
        glfwTerminate();
        return EXIT_FAILURE;
    }
    glfwMakeContextCurrent( window );

    // Set up mouse
    glfwSetMouseButtonCallback( window, mouse_button_callback );
    glfwSetCursorPosCallback( window, cursor_position_callback );
    drag_mode = false;

    // Start GLEW extension handler
    glewExperimental = GL_TRUE;
    glewInit();

    // Get version info
    const GLubyte* renderer = glGetString( GL_RENDERER ); // get renderer string
    const GLubyte* version = glGetString( GL_VERSION ); // version as a string
    printf( "Renderer: %s\n", renderer );
    printf( "OpenGL version supported %s\n", version );

    // Tell GL to only draw onto a pixel if the shape is closer to the viewer
    //glEnable( GL_DEPTH_TEST ); // enable depth-testing
    //glDepthFunc( GL_LESS ); // depth-testing interprets a smaller value as "closer"

    // Define some points (to make a square)
    float points[] = {
        // vertices   // texcoords
         0.5f,  0.5f, 1.0f, 1.0f,
         0.5f, -0.5f, 1.0f, 0.0f,
        -0.5f,  0.5f, 0.0f, 1.0f,
        -0.5f, -0.5f, 0.0f, 0.0f
    };

    // Define a place for the points to live in global memory
    //gpuErrchk( cudaMalloc( (void **)&d_points, sizeof(points) ) );

    GLuint vbo = 0;
    glGenBuffers( 1, &vbo );
    glBindBuffer( GL_ARRAY_BUFFER, vbo );
    glBufferData( GL_ARRAY_BUFFER, 16 * sizeof(float), points, GL_STATIC_DRAW );

    // Prepare a resource for CUDA interoperability
    cudaGraphicsGLRegisterBuffer( &cudaPointsResource, vbo, cudaGraphicsMapFlagsNone );

    GLuint vao = 0;
    glGenVertexArrays( 1, &vao );
    glBindVertexArray( vao );
    glBindBuffer( GL_ARRAY_BUFFER, vbo );

    glVertexAttribPointer( 0, 2, GL_FLOAT, GL_FALSE, 4*sizeof(float), NULL );
    glEnableVertexAttribArray( 0 );
    glVertexAttribPointer( 1, 2, GL_FLOAT, GL_FALSE, 4*sizeof(float), (void *)(2*sizeof(float)) );
    glEnableVertexAttribArray( 1 );

    // Texture
    GLuint tex;
    glGenTextures( 1, &tex );
    glBindTexture( GL_TEXTURE_2D, tex );

    glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT );
    glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT );
    glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST );
    glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST );

    /*
    float image[] = { 0.0f, 0.1f, 0.2f, 0.3f, 0.4f, 0.5f,
                      0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f,
                      0.2f, 0.3f, 0.4f, 0.5f, 0.6f, 0.7f,
                      0.3f, 0.4f, 0.5f, 0.6f, 0.0f, 0.8f, // <--- deliberate black pixel
                      0.4f, 0.5f, 0.6f, 0.7f, 0.8f, 0.9f,
                      0.5f, 0.6f, 0.7f, 0.8f, 0.9f, 1.0f };
     */

    w = h = 8;
    glTexImage2D( GL_TEXTURE_2D, 0, GL_R32F, w, h, 0, GL_RED, GL_FLOAT, NULL );

    glBindTexture( GL_TEXTURE_2D, 0 );

    gpuErrchk(
            cudaGraphicsGLRegisterImage(
                &cudaImageResource,
                tex,
                GL_TEXTURE_2D,
                cudaGraphicsRegisterFlagsSurfaceLoadStore
                )
            );

    gpuErrchk( cudaGraphicsMapResources( 1, &cudaImageResource, 0 ) );
    gpuErrchk( cudaGraphicsSubResourceGetMappedArray( &cuArray, cudaImageResource, 0, 0 ) );

    // CUDA Surface
    memset( &surfRes, 0, sizeof(cudaResourceDesc) );
    surfRes.resType = cudaResourceTypeArray;
    surfRes.res.array.array = cuArray;
    gpuErrchk( cudaCreateSurfaceObject( &surf, &surfRes ) );
    printf( "surfRes: (w, h) = %ld, %ld\n", surfRes.res.pitch2D.width, surfRes.res.pitch2D.height );
    //dim3 threads(w, h);
    dim3 threads(w, h);
    cudaCreateImage<<<1,threads>>>( surf, w, h );
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );

    gpuErrchk( cudaGraphicsUnmapResources( 1, &cudaImageResource, 0 ) );

    // Set up camera

    const char* vertex_shader   = loadFileContentsAsStr( "vert.shader" );
    const char* fragment_shader = loadFileContentsAsStr( "frag.shader" );

    GLuint vs = glCreateShader(GL_VERTEX_SHADER);
    glShaderSource(vs, 1, &vertex_shader, NULL);
    glCompileShader(vs);
    GLuint fs = glCreateShader(GL_FRAGMENT_SHADER);
    glShaderSource(fs, 1, &fragment_shader, NULL);
    glCompileShader(fs);

    GLuint shader_programme = glCreateProgram();
    glAttachShader(shader_programme, fs);
    glAttachShader(shader_programme, vs);
    glLinkProgram(shader_programme);

    // Set up camera
    glm::mat4 Model( 1.0f ), View( 1.0f );

    glUseProgram(shader_programme);

    GLint model = glGetUniformLocation( shader_programme, "Model" );
    glUniformMatrix4fv( model, 1, GL_FALSE, glm::value_ptr(Model) );
 
    GLint view = glGetUniformLocation( shader_programme, "View" );
    glUniformMatrix4fv( view, 1, GL_FALSE, glm::value_ptr(View) );
 
    while(!glfwWindowShouldClose(window))
    {
        // wipe the drawing surface clear
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        //glActiveTexture( GL_TEXTURE0 );
        glBindTexture( GL_TEXTURE_2D, tex );

        // draw points 0-3 from the currently bound VAO with current in-use shader
        glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);
        // update other events like input handling
        glfwPollEvents();

        // put the stuff we've been drawing onto the display
        glfwSwapBuffers(window);
    }

    //gpuErrchk( cudaFree( d_points ) );
    gpuErrchk( cudaDestroySurfaceObject( surf ) );
    gpuErrchk( cudaFreeArray( cuArray ) );

    // close GL context and any other GLFW resources
    glfwTerminate();

    return EXIT_SUCCESS;
}
