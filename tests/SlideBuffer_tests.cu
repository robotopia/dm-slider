#include <cstdlib>
#include <iostream>

#include "../SlideBuffer.h"

#define Assert(cond,failstr,count) {if (!(cond)) { cout << "[FAIL]: " << failstr << endl; return count; } count++;}

using namespace std;

int testAllocation()
{
    int nPassed = 0;

    size_t bytes = 1024;
    SlideBuffer *pSlideBuffer = NULL;
    pSlideBuffer = new SlideBuffer( 1024, ON_HOST );

    // Test 1: Construct a (CPU) SlideBuffer object
    Assert( pSlideBuffer != NULL, "Could not allocate memory for SlideBuffer", nPassed );

    // Test 2: Make sure the memory was actually allocated
    Assert( pSlideBuffer->getBuffer() != NULL, "Failed to allocate buffer memory on host", nPassed );

    // Test 3: Make sure the size of the buffer was recorded correctly
    Assert( bytes == pSlideBuffer->getSize(), "Slide buffer size incorrect", nPassed );

    delete pSlideBuffer; // Not sure how to test this

    return nPassed;
}

int main( int argc, char *argv[] )
{
    int nPassed = 0;
    nPassed += testAllocation();

    cout << "Passed " << nPassed << " tests" << endl;

    return EXIT_SUCCESS;
}
