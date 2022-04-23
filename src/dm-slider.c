#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <string.h>

// GL includes must come before "cuda_gl_interop.h"
#include <GL/glew.h>
#include <GLFW/glfw3.h>

#include <cuda_runtime.h>
#include <cuComplex.h>
#include "cudaErrorChecking.h"

#include <cuda_gl_interop.h>

#include "dm-slider.h"

#include <gtk/gtk.h>

// GLUT-related constants
#define OPEN_FILE  1

// Mouse states
static double xprev;
static double yprev;
static int drag_mode;
#define DRAG_NONE  0
#define DRAG_LEFT  1
#define DRAG_RIGHT 2

// Window states
static float windowWidth;
static float windowHeight;

#define XNORM(xpos)  ( (xpos)/windowWidth - 0.5)
#define YNORM(ypos)  (-(ypos)/windowHeight + 0.5)

struct opengl_data_t
{
    struct cudaGraphicsResource *cudaPointsResource;
    float *d_points;
    struct cudaGraphicsResource *cudaImageResource;
    float *d_image;
    cudaSurfaceObject_t surf;
    struct cudaResourceDesc surfRes;
    struct cudaArray *cuArray;
    struct cudaChannelFormatDesc channelDesc;
    int w, h;
    GLuint tex;
};

struct opengl_data_t opengl_data;

// CONTAINS CODE I STILL WANT TO RECYCLE:
/*
{
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

    // Clean up memory
    gpuErrchk( cudaFree( d_vdif ) );
    gpuErrchk( cudaFree( d_vdifData ) );
    gpuErrchk( cudaFree( d_StokesI ) );
    gpuErrchk( cudaFreeHost( vdif ) );
}
*/

void draw()
{
    // Clear the surface
    glClear( GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT );

    //glActiveTexture( GL_TEXTURE0 );
    glBindTexture( GL_TEXTURE_2D, opengl_data.tex );

    // draw points 0-3 from the currently bound VAO with current in-use shader
    glDrawArrays( GL_TRIANGLE_STRIP, 0, 4 );
}

void mouse_release_callback( GtkWidget *widget, GdkEventButton *event, gpointer data )
{
    if (!data) { } // Just to avoid compiler warning about unused parameters (delete me later)
    if (!widget)
        return;

    switch (event->button)
    {
        case 1: // Left mouse button
            drag_mode = DRAG_NONE;
            gpuErrchk( cudaGraphicsUnmapResources( 1, &(opengl_data.cudaPointsResource), 0 ) );
            break;
        case 3: // Right mouse button
            drag_mode = DRAG_NONE;
            gpuErrchk( cudaGraphicsUnmapResources( 1, &(opengl_data.cudaImageResource), 0 ) );
            break;
        default:
            break;
    }
}

void mouse_button_callback( GtkWidget *widget, GdkEventButton *event, gpointer data )
{
    if (!data) { } // Just to avoid compiler warning about unused parameters (delete me later)
    if (!widget)
        return;

    size_t size;
    switch (event->button)
    {
        case 1: // Left mouse button
            xprev = event->x;
            yprev = event->y;
            drag_mode = DRAG_LEFT;
            gpuErrchk( cudaGraphicsMapResources( 1, &(opengl_data.cudaPointsResource), 0 ) );
            gpuErrchk( cudaGraphicsResourceGetMappedPointer( (void **)&(opengl_data.d_points), &size, opengl_data.cudaPointsResource ) );
            break;
        case 3: // Right mouse button
            xprev = event->x;
            yprev = event->y;
            drag_mode = DRAG_RIGHT;
            gpuErrchk( cudaGraphicsMapResources( 1, &(opengl_data.cudaImageResource), 0 ) );
            gpuErrchk( cudaGraphicsSubResourceGetMappedArray( &(opengl_data.cuArray), opengl_data.cudaImageResource, 0, 0 ) );
            break;
        default:
            break;
    }
}

void cursor_position_callback( GtkWidget* widget, GdkEventMotion *event, gpointer data )
{
    if (!data) { }
    if (!widget)
        return;

    if (drag_mode == DRAG_LEFT)
    {
        double xpos = event->x;
        double ypos = event->y;

        float rad = atan2(YNORM(ypos),  XNORM(xpos)) -
                    atan2(YNORM(yprev), XNORM(xprev));

        cudaRotatePoints( opengl_data.d_points, rad );

        xprev = xpos;
        yprev = ypos;
    }
    else if (drag_mode == DRAG_RIGHT)
    {
        double xpos = event->x;
        double ypos = event->y;

        float dy = YNORM(ypos) - YNORM(yprev);

        cudaChangeBrightness( opengl_data.surf, opengl_data.d_image, dy, opengl_data.w, opengl_data.h );

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

static gboolean print_button_event( GtkWidget *widget, GdkEventButton *event, gpointer data )
{
    if (data) { }
    if (!widget)
        return false;

    printf( "clicked button #%d at (%lf,%lf)... or is it (%lf,%lf)?\n",
            event->button,
            event->x_root, event->y_root,
            event->x, event->y );

    return true;
}

static gboolean render( GtkGLArea *glarea, GdkGLContext *context, gpointer data )
{
    if (!data) { }
    if (!glarea || !context)
        return false;

    draw();

    gtk_widget_queue_draw( GTK_WIDGET(glarea) );

    return true;
}

static void on_glarea_realize( GtkGLArea *glarea )
{
    gtk_gl_area_make_current( GTK_GL_AREA(glarea) );
    if (gtk_gl_area_get_error( glarea ) != NULL)
        return;

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
    //gpuErrchk( cudaMalloc( (void **)&(opengl_data.d_points), sizeof(points) ) );

    GLuint vbo = 0;
    glGenBuffers( 1, &vbo );
    glBindBuffer( GL_ARRAY_BUFFER, vbo );
    glBufferData( GL_ARRAY_BUFFER, 16 * sizeof(float), points, GL_STATIC_DRAW );

    // Prepare a resource for CUDA interoperability
    cudaGraphicsGLRegisterBuffer( &(opengl_data.cudaPointsResource), vbo, cudaGraphicsMapFlagsNone );

    GLuint vao = 0;
    glGenVertexArrays( 1, &vao );
    glBindVertexArray( vao );
    glBindBuffer( GL_ARRAY_BUFFER, vbo );

    glVertexAttribPointer( 0, 2, GL_FLOAT, GL_FALSE, 4*sizeof(float), NULL );
    glEnableVertexAttribArray( 0 );
    glVertexAttribPointer( 1, 2, GL_FLOAT, GL_FALSE, 4*sizeof(float), (void *)(2*sizeof(float)) );
    glEnableVertexAttribArray( 1 );

    // Texture
    glGenTextures( 1, &(opengl_data.tex) );
    glBindTexture( GL_TEXTURE_2D, opengl_data.tex );

    glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT );
    glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT );
    glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST );
    glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST );

    opengl_data.w = 200;
    opengl_data.h = 200;
    glTexImage2D( GL_TEXTURE_2D, 0, GL_R32F, opengl_data.w, opengl_data.h, 0, GL_RED, GL_FLOAT, NULL );

    glBindTexture( GL_TEXTURE_2D, 0 );

    gpuErrchk(
            cudaGraphicsGLRegisterImage(
                &(opengl_data.cudaImageResource),
                opengl_data.tex,
                GL_TEXTURE_2D,
                cudaGraphicsRegisterFlagsSurfaceLoadStore
                )
            );

    gpuErrchk( cudaGraphicsMapResources( 1, &(opengl_data.cudaImageResource), 0 ) );
    gpuErrchk( cudaGraphicsSubResourceGetMappedArray( &(opengl_data.cuArray), opengl_data.cudaImageResource, 0, 0 ) );

    // CUDA Surface
    memset( &(opengl_data.surfRes), 0, sizeof(struct cudaResourceDesc) );
    opengl_data.surfRes.resType = cudaResourceTypeArray;
    opengl_data.surfRes.res.array.array = opengl_data.cuArray;
    gpuErrchk( cudaCreateSurfaceObject( &(opengl_data.surf), &(opengl_data.surfRes) ) );

    // Create image
    opengl_data.d_image = cudaCreateImage( opengl_data.surf, opengl_data.w, opengl_data.h );

    gpuErrchk( cudaGraphicsUnmapResources( 1, &(opengl_data.cudaImageResource), 0 ) );

    // Set up shaders

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
    glUseProgram(shader_programme);
}

int main( int argc, char *argv[] )
{
    GtkWidget *window;
    GtkWidget *vbox;
    GtkWidget *button;
    GtkWidget *glarea;

    windowWidth = 640;
    windowHeight = 480;

    gtk_init(&argc, &argv);

    window = gtk_window_new(GTK_WINDOW_TOPLEVEL);
    gtk_window_set_title(GTK_WINDOW(window), "DM Slider");
    gtk_window_set_default_size(GTK_WINDOW(window), windowWidth, windowHeight);
    gtk_container_set_border_width(GTK_CONTAINER(window), 0);
    g_signal_connect( G_OBJECT(window), "destroy",
            G_CALLBACK(gtk_main_quit), NULL );

    vbox = gtk_box_new( GTK_ORIENTATION_VERTICAL, 5 );
    glarea = gtk_gl_area_new();
    button = gtk_button_new_with_label( "Button" );
    //gtk_widget_set_tooltip_text(button, "Button widget");

    gtk_container_add( GTK_CONTAINER(window), vbox );
    gtk_container_add( GTK_CONTAINER(vbox), glarea );
    gtk_container_add( GTK_CONTAINER(vbox), button );
    gtk_box_set_child_packing( GTK_BOX(vbox), glarea, true, true, 0, GTK_PACK_START );

    gtk_widget_set_events( glarea,
            GDK_BUTTON_PRESS_MASK |
            GDK_BUTTON_RELEASE_MASK |
            GDK_BUTTON_MOTION_MASK );
    g_signal_connect( G_OBJECT(glarea), "button-press-event",
            G_CALLBACK(mouse_button_callback), NULL );
    g_signal_connect( G_OBJECT(glarea), "button-release-event",
            G_CALLBACK(mouse_release_callback), NULL );
    g_signal_connect( G_OBJECT(glarea), "motion-notify-event",
            G_CALLBACK(cursor_position_callback), NULL );
    g_signal_connect( G_OBJECT(glarea), "render",
            G_CALLBACK(render), NULL );
    g_signal_connect( G_OBJECT(glarea), "realize",
            G_CALLBACK(on_glarea_realize), NULL );

    g_signal_connect( G_OBJECT(button), "button-press-event",
            G_CALLBACK(print_button_event), NULL );

    drag_mode = false;

    gtk_widget_show_all( window );

    gtk_main();

    // Is this ever reached?
    //gpuErrchk( cudaFree( opengl_data.d_points ) );
    gpuErrchk( cudaDestroySurfaceObject( opengl_data.surf ) );
    gpuErrchk( cudaFreeArray( opengl_data.cuArray ) );
    gpuErrchk( cudaFree( opengl_data.d_image ) );

    return EXIT_SUCCESS;
}
