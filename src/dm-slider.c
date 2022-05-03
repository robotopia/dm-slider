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
#include "cohdd.h"

#include <gtk/gtk.h>

// GLUT-related constants
#define OPEN_FILE  1

// The app window
GtkWidget *window;
GtkWidget *hpaned;
GtkWidget *vbox;
GtkWidget *settingsFrame;
GtkWidget *settings_box;
GtkWidget *dynamicRangeFrame;
GtkWidget *dynamicRangeGrid;
GtkWidget *dynamicRangeLo;
GtkWidget *dynamicRangeHi;
GtkWidget *button;
GtkWidget *glarea;
//GtkWidget *glColorbar;
GtkWidget *statusbar;
guint statusbar_context_id;
GtkWidget *taperFrame;
GtkWidget *taperComboBox;
GtkWidget *stokesFrame;
GtkWidget *stokesComboBox;

GtkWidget *menubar;
GtkWidget *menu;
GtkWidget *menuitemFile;
GtkWidget *menuitemOpen;
GtkWidget *menuitemQuit;
GtkWidget *separator;

GtkAccelGroup *accel_group;


// The voltage dynamic spectrum
struct vds_t vds;

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

GtkAllocation *alloc;
static float glAreaWidth;
static float glAreaHeight;

#define XCOORD(xmousepos)  (xmousepos/glAreaWidth*(tRange[1] - tRange[0]) + tRange[0])
#define YCOORD(ymousepos)  ((1.0 - ymousepos/glAreaHeight)*(vds.hi_freq_MHz - vds.lo_freq_MHz) + vds.lo_freq_MHz)

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
    GLint tRangeLoc;
    GLint tMaxLoc;
    float tMax;
    char stokes;
};

struct opengl_data_t opengl_data;

float dynamicRange[2];
float tRange[2];

void recalcImageFromDedispersion();
void init_texture_and_surface();

void set_dynamic_range( float lo, float hi )
{
    dynamicRange[0] = lo;
    dynamicRange[1] = hi;
    glProgramUniform2fv( opengl_data.shader_program, opengl_data.dynamicRangeLoc, 1, dynamicRange );
    char loStr[32], hiStr[32];
    gtk_entry_set_text( GTK_ENTRY(dynamicRangeLo), gcvt( dynamicRange[0], 4, loStr ) );
    gtk_entry_set_text( GTK_ENTRY(dynamicRangeHi), gcvt( dynamicRange[1], 4, hiStr ) );
}

void set_visible_time_range( float tLeft, float tRight )
{
    // Assumes opengl_data.w has been set
    tRange[0]         = tLeft;
    tRange[1]         = tRight;
    glProgramUniform2fv( opengl_data.shader_program, opengl_data.tRangeLoc, 1, tRange );
}

void draw()
{
    // Clear the surface
    glClearColor( 1.0f, 1.0f, 1.0f, 0.0f );
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
            //gpuErrchk( cudaGraphicsUnmapResources( 1, &(opengl_data.cudaPointsResource), 0 ) );
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

    gtk_widget_get_allocation( widget, alloc );
    glAreaWidth = alloc->width;
    glAreaHeight = alloc->height;

    switch (event->button)
    {
        case 1: // Left mouse button
            xprev = event->x;
            yprev = event->y;
            drag_mode = DRAG_LEFT;
            //gpuErrchk( cudaGraphicsMapResources( 1, &(opengl_data.cudaPointsResource), 0 ) );
            //gpuErrchk( cudaGraphicsResourceGetMappedPointer( (void **)&(opengl_data.d_points), &size, opengl_data.cudaPointsResource ) );
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

    double xcoord = XCOORD(event->x);
    double ycoord = YCOORD(event->y);

    if (drag_mode == DRAG_LEFT)
    {
        float dx = xcoord - XCOORD(xprev);

        set_visible_time_range( tRange[0] - dx, tRange[1] - dx );

        xprev = event->x;
    }
    else if (drag_mode == DRAG_RIGHT)
    {

        float dt = xcoord - XCOORD(xprev);
        float f  = ycoord;

        vds.DM += calcDM( dt, vds.ref_freq_MHz, f );

        recalcImageFromDedispersion();

        xprev = event->x;
        yprev = event->y;
    }

    // Update the status bar with the cursor's position in world coordinates
    gtk_statusbar_pop( GTK_STATUSBAR(statusbar), statusbar_context_id );
    char loc[64];
    sprintf( loc, "[%.3f ms, %.3f MHz]\tDM = %.3f", xcoord*1e3, ycoord, vds.DM );
    gtk_statusbar_push( GTK_STATUSBAR(statusbar), statusbar_context_id, loc );
}

static gboolean mouse_scroll_callback( GtkWidget *widget, GdkEventScroll *event, gpointer data )
{
    if (!data) { }
    if (!widget)
        return false;

    gtk_widget_get_allocation( widget, alloc );
    glAreaWidth = alloc->width;
    glAreaHeight = alloc->height;

    // Convert to "world" coordinates
    double xpos = XCOORD(event->x);

    float scale_factor;
    if (event->direction == GDK_SCROLL_UP)
    {
        scale_factor = 0.2;
        tRange[0] += scale_factor*(xpos - tRange[0]);
        tRange[1] -= scale_factor*(tRange[1] - xpos);
    }
    else if (event->direction == GDK_SCROLL_DOWN)
    {
        scale_factor = 0.25;
        tRange[0] -= scale_factor*(xpos - tRange[0]);
        tRange[1] += scale_factor*(tRange[1] - xpos);
    }
    else
        return false;

    glProgramUniform2fv( opengl_data.shader_program, opengl_data.tRangeLoc, 1, tRange );

    return true;
}

int gslist_strcmp( const void *a, const void *b )
{
    return strcmp( (char *)a, (char *)b );
}

static void update_dynamic_range_callback( GtkEntry *entry, gpointer data )
{
    if (!data) { }
    if (!entry)
        return;

    set_dynamic_range(
            atof( gtk_entry_get_text( GTK_ENTRY(dynamicRangeLo) ) ),
            atof( gtk_entry_get_text( GTK_ENTRY(dynamicRangeHi) ) ) );
}

void recalcImageFromDedispersion()
{
    cudaCoherentDedispersion(
            vds.d_spectrum,
            vds.d_dedispersed_spectrum,
            vds.size,
            vds.DM,
            ctr_freq_MHz_nth_channel( &vds, 0 ),
            vds.ref_freq_MHz,
            channel_bw_MHz( &vds ),
            vds.taperType,
            vds.Np, vds.Nc, vds.Ns );
    inverseFFT( &vds );
    cudaStokes( opengl_data.d_image, vds.d_dedispersed, vds.Ns * vds.Nc, opengl_data.stokes );

    gpuErrchk( cudaGraphicsMapResources( 1, &(opengl_data.cudaImageResource), 0 ) );
    gpuErrchk( cudaGraphicsSubResourceGetMappedArray( &(opengl_data.cuArray), opengl_data.cudaImageResource, 0, 0 ) );

    cudaCopyToSurface( opengl_data.surf, opengl_data.d_image, opengl_data.w, opengl_data.h );

    gpuErrchk( cudaGraphicsUnmapResources( 1, &(opengl_data.cudaImageResource), 0 ) );
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
        // Create a vdif context
        struct vdif_context vc;

        // Get a list of filenames
        GSList *filenames;
        filenames = gtk_file_chooser_get_filenames( chooser );
        filenames = g_slist_sort( filenames, gslist_strcmp );

        // Load VDIFs
        init_vdif_context( &vc, 100 );
        add_vdif_files_to_context( &vc, filenames );
        g_slist_free( filenames );

        // Convert VDIF to a voltage dynamic spectrum
        vds_from_vdif_context( &vds, &vc );

        // Allocate memory in d_image and use it to store Stokes I data
        gpuErrchk( cudaFree( opengl_data.d_image ) );
        gpuErrchk( cudaMalloc( (void **)&opengl_data.d_image, vds.size ) );

        // Load to surface
        opengl_data.w = vds.Ns;
        opengl_data.h = vds.Nc;
        init_texture_and_surface();

        // Set the x-size of the drawing quad and the viewing area
        float tmax = opengl_data.w*vds.dt;
        glProgramUniform1f( opengl_data.shader_program, opengl_data.tMaxLoc, tmax );
        set_visible_time_range( 0.0, tmax );

        // Set the default dynamic range for VDIF files
        set_dynamic_range( -0.001, 0.01 );

        // Don't need the vdif context any more
        destroy_all_vdif_files( &vc );

        recalcImageFromDedispersion();
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
        1.0f, 1.0f, 1.0f, 1.0f,
        1.0f, 0.0f, 1.0f, 0.0f,
        0.0f, 1.0f, 0.0f, 1.0f,
        0.0f, 0.0f, 0.0f, 0.0f
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
    opengl_data.w = vds.Ns;
    opengl_data.h = vds.Nc;

    // Display title image (assumed already loaded into vds)
    size_t size = opengl_data.w * opengl_data.h * sizeof(float);
    gpuErrchk( cudaMalloc( (void **)&(opengl_data.d_image), size ) );

    init_texture_and_surface();
    recalcImageFromDedispersion();

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
    opengl_data.tRangeLoc = glGetUniformLocation( opengl_data.shader_program, "tRange" );
    opengl_data.tMaxLoc = glGetUniformLocation( opengl_data.shader_program, "tMax" );

    glProgramUniform2fv( opengl_data.shader_program, opengl_data.dynamicRangeLoc, 1, dynamicRange );
    glProgramUniform2fv( opengl_data.shader_program, opengl_data.tRangeLoc, 1, tRange );
    glProgramUniform1f( opengl_data.shader_program, opengl_data.tMaxLoc, 1.0f );
}

static void taper_combo_box_callback( GtkComboBox* widget, gpointer data )
{
    if (!data) { }
    if (!widget)
        return;

    vds.taperType = gtk_combo_box_get_active( GTK_COMBO_BOX(widget) );
    recalcImageFromDedispersion();
}

static void stokes_combo_box_callback( GtkComboBox* widget, gpointer data )
{
    if (!data) { }
    if (!widget)
        return;

    opengl_data.stokes = *gtk_combo_box_text_get_active_text( GTK_COMBO_BOX_TEXT(widget) );
    recalcImageFromDedispersion();
}

int main( int argc, char *argv[] )
{
    windowWidth = 1080;
    windowHeight = 480;

    gtk_init(&argc, &argv);

    window = gtk_window_new(GTK_WINDOW_TOPLEVEL);
    gtk_window_set_title(GTK_WINDOW(window), "DM Slider");
    gtk_window_set_default_size(GTK_WINDOW(window), windowWidth, windowHeight);
    gtk_container_set_border_width(GTK_CONTAINER(window), 0);
    g_signal_connect( G_OBJECT(window), "destroy",
            G_CALLBACK(gtk_main_quit), NULL );

    accel_group  = gtk_accel_group_new ();
    hpaned       = gtk_paned_new( GTK_ORIENTATION_HORIZONTAL );
    vbox         = gtk_box_new( GTK_ORIENTATION_VERTICAL, 5 );
    settingsFrame = gtk_frame_new( "Settings" );
    settings_box = gtk_box_new( GTK_ORIENTATION_VERTICAL, 10 );
    dynamicRangeFrame = gtk_frame_new( "Dynamic range" );
    dynamicRangeGrid  = gtk_grid_new();
    dynamicRangeLo    = gtk_entry_new();
    dynamicRangeHi    = gtk_entry_new();
    glarea       = gtk_gl_area_new();
    //glColorbar   = gtk_gl_area_new();
    statusbar    = gtk_statusbar_new();
    button       = gtk_button_new_with_label( "Auto range" );
    taperFrame   = gtk_frame_new( "Taper function" );
    stokesFrame   = gtk_frame_new( "Stokes parameter" );

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

    // Set up the taper function combo box
    taperComboBox  = gtk_combo_box_text_new();
    gtk_combo_box_text_append( GTK_COMBO_BOX_TEXT(taperComboBox), NULL, "--None--" );
    gtk_combo_box_text_append( GTK_COMBO_BOX_TEXT(taperComboBox), NULL, "Hann window" );
    gtk_combo_box_text_append( GTK_COMBO_BOX_TEXT(taperComboBox), NULL, "Welch window" );
    gtk_combo_box_set_active( GTK_COMBO_BOX(taperComboBox), TAPER_NONE );

    // Set up the stokes parameter combo box
    stokesComboBox  = gtk_combo_box_text_new();
    gtk_combo_box_text_append( GTK_COMBO_BOX_TEXT(stokesComboBox), NULL, "I" );
    gtk_combo_box_text_append( GTK_COMBO_BOX_TEXT(stokesComboBox), NULL, "Q" );
    gtk_combo_box_text_append( GTK_COMBO_BOX_TEXT(stokesComboBox), NULL, "U" );
    gtk_combo_box_text_append( GTK_COMBO_BOX_TEXT(stokesComboBox), NULL, "V" );

    // Set the initial stokes to I
    gtk_combo_box_set_active( GTK_COMBO_BOX(stokesComboBox), 0 );
    opengl_data.stokes = 'I';

    // Connect everything together
    gtk_window_add_accel_group( GTK_WINDOW(window), accel_group ); // Doesn't do anything yet
    gtk_container_add( GTK_CONTAINER(window), vbox );
    gtk_container_add( GTK_CONTAINER(vbox), menubar );
    gtk_container_add( GTK_CONTAINER(vbox), hpaned );
    gtk_container_add( GTK_CONTAINER(vbox), statusbar );
    gtk_paned_pack1( GTK_PANED(hpaned), glarea, true, true );
    gtk_paned_pack2( GTK_PANED(hpaned), settingsFrame, true, true );
    gtk_container_add( GTK_CONTAINER(settingsFrame), settings_box );
    gtk_frame_set_label_align( GTK_FRAME(settingsFrame), 0.5, 0.5 );

    gtk_container_add( GTK_CONTAINER(settings_box), dynamicRangeFrame );
    gtk_container_add( GTK_CONTAINER(dynamicRangeFrame), dynamicRangeGrid );

    gtk_grid_attach( GTK_GRID(dynamicRangeGrid), dynamicRangeLo, 0, 0, 1, 1 );
    gtk_grid_attach( GTK_GRID(dynamicRangeGrid), dynamicRangeHi, 1, 0, 1, 1 );
    gtk_grid_attach( GTK_GRID(dynamicRangeGrid), button, 0, 1, 2, 1 );
    gtk_box_set_child_packing( GTK_BOX(vbox), hpaned, true, true, 0, GTK_PACK_START );
    gtk_container_add( GTK_CONTAINER(settings_box), taperFrame );
    gtk_container_add( GTK_CONTAINER(taperFrame), taperComboBox );
    gtk_container_add( GTK_CONTAINER(settings_box), stokesFrame );
    gtk_container_add( GTK_CONTAINER(stokesFrame), stokesComboBox );

    // Set defaults and appearance
    gtk_widget_set_size_request( glarea, windowWidth - 400, -1 );
    gtk_widget_set_margin_top( GTK_WIDGET(statusbar), 0 );
    gtk_widget_set_margin_bottom( GTK_WIDGET(statusbar), 0 );
    statusbar_context_id = gtk_statusbar_get_context_id( GTK_STATUSBAR(statusbar), "Cursor position" );
    alloc = g_new( GtkAllocation, 1 );

    // Set events
    gtk_widget_set_events( glarea,
           GDK_BUTTON_PRESS_MASK |
            GDK_BUTTON_RELEASE_MASK |
            GDK_BUTTON_MOTION_MASK |
            GDK_SCROLL_MASK |
            GDK_POINTER_MOTION_MASK );
    g_signal_connect( G_OBJECT(glarea), "button-press-event",
            G_CALLBACK(mouse_button_callback), NULL );
    g_signal_connect( G_OBJECT(glarea), "button-release-event",
            G_CALLBACK(mouse_release_callback), NULL );
    g_signal_connect( G_OBJECT(glarea), "motion-notify-event",
            G_CALLBACK(cursor_position_callback), NULL );
    g_signal_connect( G_OBJECT(glarea), "scroll-event",
            G_CALLBACK(mouse_scroll_callback), NULL );
    g_signal_connect( G_OBJECT(glarea), "render",
            G_CALLBACK(render), NULL );
    g_signal_connect( G_OBJECT(glarea), "realize",
            G_CALLBACK(on_glarea_realize), NULL );
    g_signal_connect( G_OBJECT(dynamicRangeLo), "activate",
            G_CALLBACK(update_dynamic_range_callback), NULL );
    g_signal_connect( G_OBJECT(dynamicRangeHi), "activate",
            G_CALLBACK(update_dynamic_range_callback), NULL );

    g_signal_connect( G_OBJECT(menuitemOpen), "activate",
            G_CALLBACK(open_file_callback), NULL );
    g_signal_connect( G_OBJECT(menuitemQuit), "activate",
            G_CALLBACK(gtk_main_quit), NULL );
    g_signal_connect( G_OBJECT(button), "clicked",
            G_CALLBACK(print_button_event), NULL );
    g_signal_connect( G_OBJECT(taperComboBox), "changed",
            G_CALLBACK(taper_combo_box_callback), NULL );
    g_signal_connect( G_OBJECT(stokesComboBox), "changed",
            G_CALLBACK(stokes_combo_box_callback), NULL );

    drag_mode = false;

    vds_create_title( &vds );

    gtk_widget_show_all( window );
    set_dynamic_range( 0.0, 1.0 );

    float tmax = opengl_data.w*vds.dt;
    glProgramUniform1f( opengl_data.shader_program, opengl_data.tMaxLoc, tmax );
    set_visible_time_range( 0.0, tmax );

    gtk_main();

    //gpuErrchk( cudaFree( opengl_data.d_points ) );
    gpuErrchk( cudaDestroySurfaceObject( opengl_data.surf ) );
    gpuErrchk( cudaFreeArray( opengl_data.cuArray ) );
    gpuErrchk( cudaFree( opengl_data.d_image ) );

    return EXIT_SUCCESS;
}
