#include <cstdlib>
#include <iostream>

#include "../SlideBuffer.h"

#define Assert(cond,failstr,count) {if (!(cond)) { cout << "[FAIL]: " << failstr << endl; return count; } count++;}

using namespace std;

int testAllocation()
{
    int nPassed = 0;

    // Set up
    FILE *f = fopen( "testdata.dat", "r" );
    size_t bytes = 1024;
    SlideBuffer *pSlideBuffer = NULL;
    pSlideBuffer = new SlideBuffer( 1024, f );

    // Test 1: Construct a SlideBuffer object
    Assert( pSlideBuffer != NULL, "Could not allocate memory for SlideBuffer", nPassed );

    // Tests 2&3: Make sure the memory was actually allocated (1) on host and (2) on device
    Assert( pSlideBuffer->getHostBuffer() != NULL, "Failed to allocate buffer memory on host", nPassed );
    Assert( pSlideBuffer->getDeviceBuffer() != NULL, "Failed to allocate buffer memory on device", nPassed );

    // Test 3: Make sure the size of the buffer was recorded correctly
    Assert( bytes == pSlideBuffer->getSize(), "Slide buffer size incorrect", nPassed );

    // Clean up
    delete pSlideBuffer; // Not sure how to test this
    fclose( f );

    return nPassed;
}

int main( int argc, char *argv[] )
{
    int nPassed = 0;
    nPassed += testAllocation();

    cout << "Passed " << nPassed << " tests" << endl;

    return EXIT_SUCCESS;
}
