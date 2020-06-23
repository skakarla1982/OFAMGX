## Set SRC variable
set(SRC
    AmgXSolver.cpp
)

## Generate the shared library from the sources
add_library(ofamgx SHARED ${SRC})
#target_link_libraries(ofamgx ${LIBS})
