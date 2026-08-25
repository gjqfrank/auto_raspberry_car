#include <iostream>
#include <gpiod.hpp>
#include <chrono>
#include <thread>

const unsigned char wheel_one_p=27;
const unsigned char wheel_one_n=22;

const unsigned char wheel_two_p=12;
const unsigned char wheel_two_n=16;

const unsigned char wheel_three_p=5;
const unsigned char wheel_three_n=6;

const unsigned char wheel_four_p=13;
const unsigned char wheel_four_n=19;    


int main() {
    std::cout << "Hello, Motor!" << std::endl;

    // Open the GPIO chip (usually gpiochip0 on Pi 4)
    gpiod::chip chip("gpiochip0");

    // Get the line for GPIO 17
    auto One_p = chip.get_line(wheel_one_p);
    auto One_n = chip.get_line(wheel_one_n);
    auto Two_p = chip.get_line(wheel_two_p);
    auto Two_n = chip.get_line(wheel_two_n);
    auto Three_p = chip.get_line(wheel_three_p);
    auto Three_n = chip.get_line(wheel_three_n);
    auto Four_p = chip.get_line(wheel_four_p);
    auto Four_n = chip.get_line(wheel_four_n);

    // Request the line as an output with a default value of 0 (LOW)
    One_p.request({"wheel_one", gpiod::line_request::DIRECTION_OUTPUT, 0}, 0);
    One_n.request({"wheel_one", gpiod::line_request::DIRECTION_OUTPUT, 0}, 0);
    Two_p.request({"wheel_two", gpiod::line_request::DIRECTION_OUTPUT, 0}, 0);
    Two_n.request({"wheel_two", gpiod::line_request::DIRECTION_OUTPUT,0}, 0);
    Three_p.request({"wheel_three", gpiod::line_request::DIRECTION_OUTPUT, 0}, 0);
    Three_n.request({"wheel_three", gpiod::line_request::DIRECTION_OUTPUT, 0}, 0);
    Four_p.request({"wheel_four", gpiod::line_request::DIRECTION_OUTPUT, 0}, 0);
    Four_n.request({"wheel_four", gpiod::line_request::DIRECTION_OUTPUT, 0}, 0);

    std::cout << "GPIO lines for wheel one." << std::endl;
    // Set the line value to 1 (HIGH)
    One_p.set_value(1);
    One_n.set_value(0);
    std::this_thread::sleep_for(std::chrono::seconds(5));
    One_p.set_value(0);
    One_n.set_value(0);

    std::cout << "GPIO lines for wheel two." << std::endl;
    Two_p.set_value(1);
    Two_n.set_value(0);
    std::this_thread::sleep_for(std::chrono::seconds(5));
    Two_p.set_value(0);
    Two_n.set_value(0);

    std::cout << "GPIO lines for wheel three." << std::endl;
    Three_p.set_value(1);
    Three_n.set_value(0);
    std::this_thread::sleep_for(std::chrono::seconds(5));
    Three_p.set_value(0);
    Three_n.set_value(0);

    std::cout << "GPIO lines for wheel four." << std::endl;
    Four_p.set_value(1);
    Four_n.set_value(0);
    std::this_thread::sleep_for(std::chrono::seconds(5));
    Four_p.set_value(0);
    Four_n.set_value(0);

    std::cout << "Done." << std::endl;

    return 0;
}
