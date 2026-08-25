cmake_minimum_required(VERSION 3.0 FATAL_ERROR)

project(YoloFastestV2)

macro(use_cxx11)
  if (CMAKE_VERSION VERSION_LESS "3.1")
    if (CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
      set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17")
    endif ()
  else ()
    set (CMAKE_CXX_STANDARD 17)
  endif ()
endmacro(use_cxx11)

find_package(OpenCV REQUIRED)                              


set(SOURCES mainFV2.cpp yolo-fastestv2.cpp)
set(SOURCES_IMG mainImg.cpp yolo-fastestv2.cpp)
set(SOURCES_CAMERA mainCamera.cpp yolo-fastestv2.cpp)
set(SOURCES_LANE mainLane.cpp yolo-fastestv2.cpp)
set(SOURCES_LANETWO mainLaneTwo.cpp yolo-fastestv2.cpp)


set(SOURCES_MOTOR mainMotor.cpp yolo-fastestv2.cpp)
set(SOURCES_LANETEXT mainLaneText.cpp yolo-fastestv2.cpp)

set(EXTRA_LIBS ${OpenCV_LIBS} /usr/local/lib/ncnn/libncnn.a -lgpiodcxx)

set(EXECUTABLE_OUTPUT_PATH "../")

include_directories(include ${OpenCV_INCLUDE_DIRS} /usr/local/include/ncnn)

add_link_options(-fopenmp -s -ldl -lpthread -lgomp -rdynamic)


add_executable(YoloFastestV2 ${SOURCES})
add_executable(mainImg ${SOURCES_IMG})
add_executable(mainCamera ${SOURCES_CAMERA})

add_executable(mainLane ${SOURCES_LANE})
add_executable(mainLaneTwo ${SOURCES_LANETWO})


add_executable(mainMotor ${SOURCES_MOTOR})
add_executable(mainLaneText ${SOURCES_LANETEXT})
add_executable(testPwm testPwm.cpp)
add_executable(testpPwm testpPwm.cpp)

add_executable(zebra_crossing zebra_crossing.cpp yolo-fastestv2.cpp)
add_executable(windv3 windv3.cpp yolo-fastestv2.cpp)
add_executable(windv2 windv2.cpp yolo-fastestv2.cpp)
add_executable(rectraffic rectraffic.cpp)



add_executable(linechaser linechaser.cpp yolo-fastestv2.cpp)
add_executable(linefollower linefollower.cpp yolo-fastestv2.cpp)

add_executable(lineround lineround.cpp yolo-fastestv2.cpp)
add_executable(pwmstr pwmstr.cpp yolo-fastestv2.cpp)

add_executable(strv1 strv1.cpp yolo-fastestv2.cpp)

add_executable(engine engine.cpp yolo-fastestv2.cpp)
add_executable(05281927 05281927.cpp yolo-fastestv2.cpp)
add_executable(pwmnoncnn pwmnoncnn.cpp yolo-fastestv2.cpp)
add_executable(eyeon eyeon.cpp yolo-fastestv2.cpp)
add_executable(redandgreen redandgreen.cpp yolo-fastestv2.cpp)
add_executable(finalv1 finalv1.cpp yolo-fastestv2.cpp)
add_executable(testwheels testwheels.cpp)
 
target_link_libraries(YoloFastestV2 ${EXTRA_LIBS})
target_link_libraries(mainImg ${EXTRA_LIBS})
target_link_libraries(mainCamera ${EXTRA_LIBS})
target_link_libraries(mainLane ${EXTRA_LIBS})
target_link_libraries(mainLaneTwo ${EXTRA_LIBS})
target_link_libraries(mainLaneText ${EXTRA_LIBS})
target_link_libraries(mainMotor ${EXTRA_LIBS})
target_link_libraries(testPwm ${EXTRA_LIBS})
target_link_libraries(testpPwm ${EXTRA_LIBS} -lpigpiod_if2 -lpigpio -lrt)
target_link_libraries(zebra_crossing ${EXTRA_LIBS} -lpigpiod_if2 -lpigpio -lrt)
target_link_libraries(windv3 ${EXTRA_LIBS} -lpigpiod_if2 -lpigpio -lrt)
target_link_libraries(windv2 ${EXTRA_LIBS} -lpigpiod_if2 -lpigpio -lrt)
target_link_libraries(rectraffic ${EXTRA_LIBS})


target_link_libraries(linechaser ${EXTRA_LIBS})
target_link_libraries(linefollower ${EXTRA_LIBS} -lgpiod)

target_link_libraries(lineround ${EXTRA_LIBS} -lgpiod)
target_link_libraries(pwmstr ${EXTRA_LIBS} -lpigpiod_if2 -lpigpio -lrt)
target_link_libraries(strv1 ${EXTRA_LIBS} -lpigpiod_if2 -lpigpio -lrt)

target_link_libraries(engine ${EXTRA_LIBS} -lpigpiod_if2 -lpigpio -lrt)
target_link_libraries(05281927 ${EXTRA_LIBS} -lpigpiod_if2 -lpigpio -lrt)
target_link_libraries(pwmnoncnn ${EXTRA_LIBS} -lpigpiod_if2 -lpigpio -lrt)
target_link_libraries(eyeon ${EXTRA_LIBS} -lpigpiod_if2 -lpigpio -lrt)
target_link_libraries(redandgreen ${EXTRA_LIBS} -lpigpiod_if2 -lpigpio -lrt)
target_link_libraries(finalv1 ${EXTRA_LIBS} -lpigpiod_if2 -lpigpio -lrt)
target_link_libraries(testwheels ${EXTRA_LIBS} -lpigpiod_if2 -lpigpio -lrt)