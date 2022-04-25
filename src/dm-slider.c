#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <string.h>

// GL includes must come before "cuda_gl_interop.h"
#include <GL/glew.h>

#include <cuda_runtime.h>
#include <cuComplex.h>
#include "cudaErrorChecking.h"

#include <cuda_gl_interop.h>

#include "dm-slider.h"
#include "ascii_header.h"

#include <gtk/gtk.h>

// GLUT-related constants
#define OPEN_FILE  1

// The app window
GtkWidget *window;

// the VDIF context
struct vdif_context vc;

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
    GLuint shader_program;
    GLint dynamicRangeLoc;
};

struct opengl_data_t opengl_data;

float dynamicRange[] = { 0.0, 1.0 };

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

void init_texture_and_surface();

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

        dynamicRange[0] += dy;
        dynamicRange[1] += dy;
        glProgramUniform2fv( opengl_data.shader_program, opengl_data.dynamicRangeLoc, 1, dynamicRange );

        xprev = xpos;
        yprev = ypos;
    }
}

int gslist_strcmp( const void *a, const void *b )
{
    return strcmp( (char *)a, (char *)b );
}

static gboolean open_file_callback( GtkWidget *widget, gpointer data )
{
    if (!data) { }
    if (!widget)
        return false;

    GtkWidget *dialog;
    GtkFileChooserAction action = GTK_FILE_CHOOSER_ACTION_OPEN;
    gint res;

    dialog = gtk_file_chooser_dialog_new( "Open File",
            GTK_WINDOW(window),
            action,
            "_Cancel",
            GTK_RESPONSE_CANCEL,
            "_Open",
            GTK_RESPONSE_ACCEPT,
            NULL );

    GtkFileChooser *chooser = GTK_FILE_CHOOSER( dialog );
    gtk_file_chooser_set_select_multiple ( chooser, true );
    res = gtk_dialog_run( GTK_DIALOG(dialog) );
    if (res == GTK_RESPONSE_ACCEPT)
    {
        // Get rid of the previous lot
        destroy_all_vdif_files( &vc );

        GSList *filenames;
        filenames = gtk_file_chooser_get_filenames( chooser );
        filenames = g_slist_sort( filenames, gslist_strcmp );

        // Load VDIFs
        init_vdif_context( &vc, 8, 1024 );
        add_vdif_files_to_context( &vc, filenames );

        GSList *iter;
        struct vdif_file *vf;
        for (iter = vc.channels; iter != NULL; iter = iter->next)
        {
            vf = (struct vdif_file *)iter->data;
            printf( "%s:\n\t%f MHz\n\t%s\n", vf->hdrfile, vf->ctr_freq_MHz, vf->datafile );
        }

        // Allocate memory in d_image and use it to store Stokes I data
        gpuErrchk( cudaFree( opengl_data.d_image ) );
        gpuErrchk( cudaMalloc( (void **)&opengl_data.d_image, vc.size ) );
        cudaStokesI( opengl_data.d_image, vc.d_data, vc.ndual_pol_samples );

        // Load to surface
        int nchans = g_slist_length( vc.channels );
        opengl_data.w = vc.ndual_pol_samples / nchans;
        opengl_data.h = nchans;
        init_texture_and_surface();

        g_slist_free( filenames );
// DEBUG
float *image;
size_t size = opengl_data.w * opengl_data.h * sizeof(float);
gpuErrchk( cudaMallocHost( (void **)&image, size ) );
gpuErrchk( cudaMemcpy( image, opengl_data.d_image, size, cudaMemcpyDeviceToHost ) );
FILE *f = fopen( "foo.txt", "w" );
int x, y;
for (x = 0; x < opengl_data.h; x++)
for (y = 0; y < opengl_data.w; y++)
{
    fprintf( f, "%d %d %f\n", x, y, image[x*opengl_data.w + y] );
}
fclose(f);

    }

    gtk_widget_destroy( dialog );

    return true;
}

static gboolean print_button_event( GtkWidget *widget, gpointer data )
{
    if (data) { }
    if (!widget)
        return false;

    printf( "Button clicked\n" );

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

void init_texture_and_surface()
{
    // Assumes opengl_data member variables w, h, and d_image are set

    glDeleteTextures( 1, &(opengl_data.tex) );
    glGenTextures( 1, &(opengl_data.tex) );
    glBindTexture( GL_TEXTURE_2D, opengl_data.tex );

    glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT );
    glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT );
    glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST );
    glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST );

    gpuErrchk( cudaDestroySurfaceObject( opengl_data.surf ) );
    gpuErrchk( cudaFreeArray( opengl_data.cuArray ) );

    glBindTexture( GL_TEXTURE_2D, opengl_data.tex );

    glTexImage2D( GL_TEXTURE_2D, 0, GL_R32F, opengl_data.w, opengl_data.h, 0, GL_RED, GL_FLOAT, NULL );

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

    fprintf( stderr, "w, h = %d, %d\n", opengl_data.w, opengl_data.h );
    cudaCopyToSurface( opengl_data.surf, opengl_data.d_image, opengl_data.w, opengl_data.h );

    gpuErrchk( cudaGraphicsUnmapResources( 1, &(opengl_data.cudaImageResource), 0 ) );

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
        0.5f,   0.5f, 1.0f, 1.0f,
        0.5f,  -0.5f, 1.0f, 0.0f,
        -0.5f,  0.5f, 0.0f, 1.0f,
        -0.5f, -0.5f, 0.0f, 0.0f
    };

    // Define a place for the points to live in global memory
    //gpuErrchk( cudaMalloc( (void **)&(opengl_data.d_points), sizeof(points) ) );

    GLuint vbo;
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

    // Set up initial texture and surface
    opengl_data.w = 10;
    opengl_data.h = 10;

    // Create dummy image
    size_t size = opengl_data.w * opengl_data.h * sizeof(float);
    gpuErrchk( cudaMalloc( (void **)&(opengl_data.d_image), size ) );
    cudaCreateImage( opengl_data.d_image, opengl_data.w, opengl_data.h );

    init_texture_and_surface();

    // Set up shaders

    const char* vertex_shader   = load_file_contents_as_str( "vert.shader" );
    const char* fragment_shader = load_file_contents_as_str( "frag.shader" );
    if (!vertex_shader || !fragment_shader)
    {
        fprintf( stderr, "ERROR: Couldn't load shaders from file\n" );
        exit(EXIT_FAILURE);
    }

    GLuint vs = glCreateShader(GL_VERTEX_SHADER);
    glShaderSource(vs, 1, &vertex_shader, NULL);
    glCompileShader(vs);
    GLuint fs = glCreateShader(GL_FRAGMENT_SHADER);
    glShaderSource(fs, 1, &fragment_shader, NULL);
    glCompileShader(fs);

    opengl_data.shader_program = glCreateProgram();
    glAttachShader(opengl_data.shader_program, fs);
    glAttachShader(opengl_data.shader_program, vs);
    glLinkProgram(opengl_data.shader_program);
    glUseProgram(opengl_data.shader_program);

    opengl_data.dynamicRangeLoc = glGetUniformLocation( opengl_data.shader_program, "DynamicRange" );
    glProgramUniform2fv( opengl_data.shader_program, opengl_data.dynamicRangeLoc, 1, dynamicRange );
}

int main( int argc, char *argv[] )
{
    GtkWidget *vbox;
    GtkWidget *button;
    GtkWidget *glarea;
    GtkWidget *menubar;
    GtkWidget *menu;
    GtkWidget *menuitemFile;
    GtkWidget *menuitemOpen;
    GtkWidget *menuitemQuit;
    GtkWidget *separator;
    GtkAccelGroup *accel_group;

    windowWidth = 640;
    windowHeight = 480;

    gtk_init(&argc, &argv);

    window = gtk_window_new(GTK_WINDOW_TOPLEVEL);
    gtk_window_set_title(GTK_WINDOW(window), "DM Slider");
    gtk_window_set_default_size(GTK_WINDOW(window), windowWidth, windowHeight);
    gtk_container_set_border_width(GTK_CONTAINER(window), 0);
    g_signal_connect( G_OBJECT(window), "destroy",
            G_CALLBACK(gtk_main_quit), NULL );

    accel_group = gtk_accel_group_new ();
    vbox        = gtk_box_new( GTK_ORIENTATION_VERTICAL, 5 );
    glarea      = gtk_gl_area_new();
    button      = gtk_button_new_with_label( "Button" );
    //gtk_widget_set_tooltip_text(button, "Button widget");

    // Add menu items
    menubar      = gtk_menu_bar_new();
    menu         = gtk_menu_new();
    menuitemFile = gtk_menu_item_new_with_label( "File" );
    menuitemOpen = gtk_menu_item_new_with_label( "Open" );
    separator    = gtk_separator_menu_item_new();
    menuitemQuit = gtk_menu_item_new_with_label( "Quit" );

    gtk_menu_shell_append( GTK_MENU_SHELL(menu), menuitemOpen );
    gtk_menu_shell_append( GTK_MENU_SHELL(menu), separator );
    gtk_menu_shell_append( GTK_MENU_SHELL(menu), menuitemQuit );
    gtk_menu_item_set_submenu( GTK_MENU_ITEM(menuitemFile), menu );
    gtk_menu_shell_append( GTK_MENU_SHELL(menubar), menuitemFile );

    // Connect everything together
    gtk_window_add_accel_group( GTK_WINDOW(window), accel_group ); // Doesn't do anything yet
    gtk_container_add( GTK_CONTAINER(window), vbox );
    gtk_container_add( GTK_CONTAINER(vbox), menubar );
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

    g_signal_connect( G_OBJECT(menuitemOpen), "activate",
            G_CALLBACK(open_file_callback), NULL );
    g_signal_connect( G_OBJECT(menuitemQuit), "activate",
            G_CALLBACK(gtk_main_quit), NULL );
    g_signal_connect( G_OBJECT(button), "clicked",
            G_CALLBACK(print_button_event), NULL );

    drag_mode = false;

    gtk_widget_show_all( window );

    gtk_main();

    //gpuErrchk( cudaFree( opengl_data.d_points ) );
    gpuErrchk( cudaDestroySurfaceObject( opengl_data.surf ) );
    gpuErrchk( cudaFreeArray( opengl_data.cuArray ) );
    gpuErrchk( cudaFree( opengl_data.d_image ) );

    destroy_all_vdif_files( &vc );

    return EXIT_SUCCESS;
}
