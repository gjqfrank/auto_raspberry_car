#include <cstdlib>
#include <iostream>
#include <chrono>
#include <pigpiod_if2.h>
#include <thread>

/* pwm 运行成功 ！*/


#define LF_FWD  1
#define LF_BWD  7
#define LF_PWM  12

#define LB_FWD  24
#define LB_BWD  23
#define LB_PWM  18

#define RF_FWD  6
#define RF_BWD  5
#define RF_PWM  13

#define RB_FWD  21
#define RB_BWD  20
#define RB_PWM  19

int freq_ch0 =100;// 2 kHz for Channel 0
int default_duty = 400000; // 25% duty cycle (0-1000000), 50% duty cycle would be 500000
#define PWM_DUTY_400000 400000 // 35% duty cycle for turning

/* 350000, 450000 作为两轮的差值，看起来比较适合。300000之下的话，轮子转不起来。 */

using namespace std;


int main(int argc, char* argv[]) {

    cout << "This test is for pigpio PWM, not gpiod." << endl;

    // Connect to the pigpiod daemon (localhost, default port)
    int pi = pigpio_start(NULL, NULL);
    if (pi < 0) {
        std::cerr << "Failed to connect to pigpiod daemon." << std::endl;
        return 1;
    }

    set_mode(pi, LF_PWM, PI_OUTPUT);
    set_mode(pi, LF_FWD, PI_OUTPUT);
    set_mode(pi, LF_BWD, PI_OUTPUT);

    set_mode(pi, LB_PWM, PI_OUTPUT);
    set_mode(pi, LB_FWD, PI_OUTPUT);
    set_mode(pi, LB_BWD, PI_OUTPUT);

    set_mode(pi, RF_PWM, PI_OUTPUT);
    set_mode(pi, RF_FWD, PI_OUTPUT);
    set_mode(pi, RF_BWD, PI_OUTPUT);

    set_mode(pi, RB_PWM, PI_OUTPUT);
    set_mode(pi, RB_FWD, PI_OUTPUT);
    set_mode(pi, RB_BWD, PI_OUTPUT);

    gpio_write(pi, LF_FWD, 1); // Set forward pin high
    gpio_write(pi, LF_BWD, 0); // Set backward pin low
    gpio_write(pi, LB_FWD, 1); // Set forward pin high
    gpio_write(pi, LB_BWD, 0); // Set backward pin low
    gpio_write(pi, RF_FWD, 1); // Set forward pin high
    gpio_write(pi, RF_BWD, 0); // Set backward pin low
    gpio_write(pi, RB_FWD, 1); // Set forward pin high
    gpio_write(pi, RB_BWD, 0); // Set backward pin low

    // Start PWM with initial duty cycles (range is 0 to 1000000 for 100%)
    // hardware_PWM(pi, gpio, frequency, duty_cycle)
    hardware_PWM(pi, LF_PWM, freq_ch0, 750000);  // % duty cycle
    hardware_PWM(pi, LB_PWM, freq_ch0, 750000);  // % duty cycle
    hardware_PWM(pi, RF_PWM, freq_ch0, 300000);  // % duty cycle
    hardware_PWM(pi, RB_PWM, freq_ch0, 300000);  // % duty cycle

    std::this_thread::sleep_for(std::chrono::milliseconds(3000));; // Run for 5 seconds

    gpio_write(pi, LF_FWD, 0); // Stop the motor
    gpio_write(pi, LF_BWD, 0);
    hardware_PWM(pi, LF_PWM, 0, 0);  // 0% duty cycle

    gpio_write(pi, LB_FWD, 0); // Stop the motor
    gpio_write(pi, LB_BWD, 0);
    hardware_PWM(pi, LB_PWM, 0, 0);  // 0% duty cycle

    gpio_write(pi, RF_FWD, 0); // Stop the motor
    gpio_write(pi, RF_BWD, 0);
    hardware_PWM(pi, RF_PWM, 0, 0);  // 0%

    gpio_write(pi, RB_FWD, 0); // Stop the motor
    gpio_write(pi, RB_BWD, 0);
    hardware_PWM(pi, RB_PWM, 0, 0);  // 0%

    cout << "stop pwm" << endl;

    // Disconnect from the daemon
    pigpio_stop(pi);




    // const unsigned int gpio_pin = (argc > 1) ? static_cast<unsigned int>(std::strtoul(argv[1], nullptr, 10)) : 18;
    // const unsigned int frequency_hz = (argc > 2) ? static_cast<unsigned int>(std::strtoul(argv[2], nullptr, 10)) : 1000;
    // const unsigned int duty_percent = (argc > 3) ? static_cast<unsigned int>(std::strtoul(argv[3], nullptr, 10)) : 50;

    // if (frequency_hz == 0 || duty_percent > 100) {
    //     std::cerr << "Usage: ./testpPwm [gpio_pin] [frequency_hz>0] [duty_percent 0-100]" << std::endl;
    //     return 1;
    // }

    // if (gpioInitialise() < 0) {
    //     std::cerr << "pigpio init failed" << std::endl;
    //     return 1;
    // }

    // gpioSetMode(gpio_pin, PI_OUTPUT);
    // gpioSetPWMfrequency(gpio_pin, static_cast<int>(frequency_hz));

    // const unsigned int duty_0_255 = (duty_percent * 255) / 100;
    // gpioPWM(gpio_pin, static_cast<unsigned int>(duty_0_255));

    // std::cout << "PWM active on GPIO " << gpio_pin
    //           << " | frequency=" << frequency_hz << "Hz"
    //           << " | duty=" << duty_percent << "%" << std::endl;
    // std::cout << "Press Enter to stop..." << std::endl;
    // std::cin.get();

    // gpioPWM(gpio_pin, 0);
    // gpioTerminate();
    // std::cout << "PWM stopped." << std::endl;

    return 0;
}