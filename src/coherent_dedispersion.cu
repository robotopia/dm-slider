#include <cuda_runtime.h>
#include <cuComplex.h>
#include "../src/cudaErrorChecking.h"

//#include <GL/gl.h>
//#include <GL/glut.h>

#include <GL/glew.h>
#define GLFW_DLL
#include <GLFW/glfw3.h>

#include <stdlib.h>
#include <stdio.h>

// GLUT-related constants
#define OPEN_FILE  1

// Mouse position
static double xpos;
static double ypos;

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

// Clears the current window and draws a triangle.
void display()
{
    // Set every pixel in the frame buffer to the current clear color.
    glClear(GL_COLOR_BUFFER_BIT);

    // Drawing is done by specifying a sequence of vertices.  The way these
    // vertices are connected (or not connected) depends on the argument to
    // glBegin.  GL_POLYGON constructs a filled polygon.
    glBegin(GL_POLYGON);
    {
        glColor3f(1, 0, 0); glVertex3f(-0.6, -0.75, 0.5);
        glColor3f(0, 1, 0); glVertex3f(0.6, -0.75, 0);
        glColor3f(0, 0, 1); glVertex3f(0, 0.75, 0);
    }
    glEnd();

    // Flush drawing command buffer to make drawing happen as soon as possible.
    glFlush();
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
    if (button == GLFW_MOUSE_BUTTON_LEFT && action == GLFW_PRESS)
    {
        glfwGetCursorPos( window, &xpos, &ypos );
        fprintf( stderr, "(x, y) = (%lf, %lf)\n", xpos, ypos );
    }
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

    GLFWwindow* window = glfwCreateWindow( 640, 480, "Hello Triangle", NULL, NULL );
    if (!window)
    {
        fprintf(stderr, "ERROR: could not open window with GLFW3\n");
        glfwTerminate();
        return EXIT_FAILURE;
    }
    glfwMakeContextCurrent( window );
    glfwSetMouseButtonCallback( window, mouse_button_callback );

    // Start GLEW extension handler
    glewExperimental = GL_TRUE;
    glewInit();

    // Get version info
    const GLubyte* renderer = glGetString( GL_RENDERER ); // get renderer string
    const GLubyte* version = glGetString( GL_VERSION ); // version as a string
    printf( "Renderer: %s\n", renderer );
    printf( "OpenGL version supported %s\n", version );

    // Tell GL to only draw onto a pixel if the shape is closer to the viewer
    glEnable( GL_DEPTH_TEST ); // enable depth-testing
    glDepthFunc( GL_LESS ); // depth-testing interprets a smaller value as "closer"

    // Define a triangle
    float points[] = {
        0.5f,  0.5f,  0.0f,
        0.5f, -0.5f,  0.0f,
        -0.5f, 0.5f,  0.0f,
        -0.5f, -0.5f,  0.0f
    };

    GLuint vbo = 0;
    glGenBuffers( 1, &vbo );
    glBindBuffer( GL_ARRAY_BUFFER, vbo );
    glBufferData( GL_ARRAY_BUFFER, 12 * sizeof(float), points, GL_STATIC_DRAW );

    GLuint vao = 0;
    glGenVertexArrays( 1, &vao );
    glBindVertexArray( vao );
    glEnableVertexAttribArray( 0 );
    glBindBuffer( GL_ARRAY_BUFFER, vbo );
    glVertexAttribPointer( 0, 3, GL_FLOAT, GL_FALSE, 0, NULL );

    const char* vertex_shader =
        "#version 400\n"
        "in vec3 vp;"
        "void main() {"
        "  gl_Position = vec4(vp, 1.0);"
        "}";

    const char* fragment_shader =
        "#version 400\n"
        "out vec4 frag_colour;"
        "void main() {"
        "  frag_colour = vec4(0.5, 0.0, 0.5, 1.0);"
        "}";

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

    while(!glfwWindowShouldClose(window))
    {
        // wipe the drawing surface clear
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        glUseProgram(shader_programme);
        glBindVertexArray(vao);
        // draw points 0-3 from the currently bound VAO with current in-use shader
        glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);
        // update other events like input handling
        glfwPollEvents();
        // put the stuff we've been drawing onto the display
        glfwSwapBuffers(window);
    }

    // close GL context and any other GLFW resources
    glfwTerminate();

    return EXIT_SUCCESS;
}
