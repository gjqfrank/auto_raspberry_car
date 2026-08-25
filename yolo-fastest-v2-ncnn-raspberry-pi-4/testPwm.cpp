#include <chrono>
#include <cstdlib>
#include <iostream>
#include <thread>

#include <gpiod.hpp>

int main(int argc, char* argv[]) {
    const unsigned int gpio_pin = (argc > 1) ? static_cast<unsigned int>(std::strtoul(argv[1], nullptr, 10)) : 18;
    const unsigned int frequency_hz = (argc > 2) ? static_cast<unsigned int>(std::strtoul(argv[2], nullptr, 10)) : 100;
    const unsigned int duty_percent = (argc > 3) ? static_cast<unsigned int>(std::strtoul(argv[3], nullptr, 10)) : 50;
    const unsigned int duration_seconds = (argc > 4) ? static_cast<unsigned int>(std::strtoul(argv[4], nullptr, 10)) : 5;

    if (frequency_hz == 0 || duty_percent > 100 || duration_seconds == 0) {
        std::cerr << "Usage: ./testPwm [gpio_pin] [frequency_hz>0] [duty_percent 0-100] [duration_seconds>0]" << std::endl;
        return 1;
    }

    const auto period_us = std::chrono::microseconds(1000000 / frequency_hz);
    const auto high_us = std::chrono::microseconds((period_us.count() * duty_percent) / 100);
    const auto low_us = std::chrono::microseconds(period_us.count() - high_us.count());
    const auto end_time = std::chrono::steady_clock::now() + std::chrono::seconds(duration_seconds);

    try {
        gpiod::chip chip("gpiochip0");
        auto pwm_line = chip.get_line(gpio_pin);
        pwm_line.request({"test-pwm", gpiod::line_request::DIRECTION_OUTPUT, 0}, 0);

        std::cout << "PWM test started on GPIO " << gpio_pin
                  << " | f=" << frequency_hz << "Hz"
                  << " | duty=" << duty_percent << "%"
                  << " | duration=" << duration_seconds << "s" << std::endl;

        while (std::chrono::steady_clock::now() < end_time) {
            if (high_us.count() > 0) {
                pwm_line.set_value(1);
                std::this_thread::sleep_for(high_us);
            }
            if (low_us.count() > 0) {
                pwm_line.set_value(0);
                std::this_thread::sleep_for(low_us);
            }
        }

        pwm_line.set_value(0);
        std::cout << "PWM test finished." << std::endl;
    } catch (const std::exception& ex) {
        std::cerr << "GPIO error: " << ex.what() << std::endl;
        return 1;
    }

    return 0;
}