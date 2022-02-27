#include <cstdlib>
#include <iostream>

#include "../src/VDIFBuffer.h"

#define Assert(cond,failstr,count) {if (!(cond)) { cout << "[FAIL]: " << failstr << endl; return count; } count++;}

using namespace std;

// From https://stackoverflow.com/questions/2082743/c-equivalent-to-fstreams-peek
int fpeek(FILE *stream)
{
    int c;

    c = fgetc(stream);
    ungetc(c, stream);

    return c;
}

int testAllocation()
{
    int nPassed = 0;

    // Set up
    size_t bytes = 1024;
    VDIFBuffer *pVDIFBuffer = NULL;
    pVDIFBuffer = new VDIFBuffer( bytes );

    // Test 1: Construct a VDIFBuffer object
    Assert( pVDIFBuffer != NULL, "Could not allocate memory for VDIFBuffer", nPassed );

    // Tests 2&3: Make sure the memory was actually allocated (1) on host and (2) on device
    Assert( pVDIFBuffer->getHostBuffer() != NULL, "Failed to allocate buffer memory on host", nPassed );
    Assert( pVDIFBuffer->getDeviceBuffer() != NULL, "Failed to allocate buffer memory on device", nPassed );

    // Test 4: Make sure the size of the buffer was recorded correctly
    Assert( bytes == pVDIFBuffer->getSize(), "Slide buffer size incorrect", nPassed );

    // Clean up
    delete pVDIFBuffer; // Not sure how to test this

    return nPassed;
}

int testSlide()
{
    int nPassed = 0;

    // Set up
    size_t bytes = 64;
    VDIFBuffer myVDIFBuffer( bytes, "testdata.vdif" );
    myVDIFBuffer.fillBuffer();

    // Test 1: Make sure the data copied ok
    myVDIFBuffer.pullBuffer();
    char *buffer = (char *)myVDIFBuffer.getHostBuffer();
    Assert( buffer[0] == '1' && buffer[1] == '2', "Buffer contents not copied to/from device correctly", nPassed );

    // Test 2: Try a slide by 1 byte
    myVDIFBuffer.slideAndRead(1);
    myVDIFBuffer.pullBuffer();
    Assert( buffer[0] == '2' && buffer[1] == '2', "slideAndRead() did not slide correctly (1 byte test)", nPassed );

    // Test 3: Try a slide by 10 bytes
    myVDIFBuffer.slideAndRead(10);
    myVDIFBuffer.pullBuffer();
    Assert( buffer[10] == 'b' && buffer[11] == 'b', "slideAndRead() did not slide correctly (10 byte test)", nPassed );

    // Test 4: Try a slide by 100 bytes
    myVDIFBuffer.slideAndRead(100);
    myVDIFBuffer.pullBuffer();
    Assert( buffer[0] == 'M' && buffer[bytes-1] == 'M', "slideAndRead() did not slide correctly (100 byte test)", nPassed );

    // Test 5: Try a slide by -1 bytes
    myVDIFBuffer.slideAndRead(-1);
    myVDIFBuffer.pullBuffer();
    Assert( buffer[0] == 'M' && buffer[bytes-1] == 'L', "slideAndRead() did not slide correctly (-1 byte test)", nPassed );

    // Test 6: Try a slide by -5 bytes
    myVDIFBuffer.slideAndRead(-5);
    myVDIFBuffer.pullBuffer();
    Assert( buffer[bytes-7] == 'G' && buffer[bytes-6] == 'G', "slideAndRead() did not slide correctly (-5 byte test)", nPassed );

    // Test 7: Try a slide by -100 bytes
    myVDIFBuffer.slideAndRead(-100);
    myVDIFBuffer.pullBuffer();
    Assert( buffer[0] == '6' && buffer[bytes-1] == '6', "slideAndRead() did not slide correctly (-100 byte test)", nPassed );

    // Test 8: Try a slide by -50 bytes (should just go to beginning of file)
    myVDIFBuffer.slideAndRead(-50);
    myVDIFBuffer.pullBuffer();
    Assert( buffer[bytes-5] == '1' && buffer[bytes-1] == '5', "slideAndRead() did not slide correctly (-50 byte test)", nPassed );

/*
// Some debug code, if needed
printf( "%c, ", fpeek(f) );
printf( "%c\n", fpeek(f) );
for (int i = 0; i < bytes; i++)
    printf("%c", buffer[i]);
printf("\n");
*/

    return nPassed;
}

int main( int argc, char *argv[] )
{
    cout << "testAllocation: " << testAllocation() << "/4 tests passed" << endl;
    cout << "testSlide:      " << testSlide()      << "/8 tests passed" << endl;

    return EXIT_SUCCESS;
}
