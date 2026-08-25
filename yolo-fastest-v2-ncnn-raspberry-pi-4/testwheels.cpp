#include <pigpiod_if2.h>
#include <iostream>
#include <unistd.h>
#include <termios.h>
#include <fcntl.h>

// ==================== 完全沿用你原来的引脚定义 ====================
#define LF_FWD      1
#define LF_BWD      7
#define LF_PWM      12

#define LB_FWD      24
#define LB_BWD      23
#define LB_PWM      18

#define RF_FWD      6
#define RF_BWD      5
#define RF_PWM      13

#define RB_FWD      21
#define RB_BWD      20
#define RB_PWM      19

// PWM 参数
const int PWM_FREQ = 100;
const int TEST_DUTY = 300000; // 测试用占空比（适中）

int pi;

// 键盘读取（无需回车）
int kbhit(void) {
    struct termios oldt, newt;
    int ch;
    int oldf;
    tcgetattr(STDIN_FILENO, &oldt);
    newt = oldt;
    newt.c_lflag &= ~(ICANON | ECHO);
    tcsetattr(STDIN_FILENO, TCSANOW, &newt);
    oldf = fcntl(STDIN_FILENO, F_GETFL, 0);
    fcntl(STDIN_FILENO, F_SETFL, oldf | O_NONBLOCK);
    ch = getchar();
    tcsetattr(STDIN_FILENO, TCSANOW, &oldt);
    fcntl(STDIN_FILENO, F_SETFL, oldf);
    if(ch != EOF) return ch;
    return 0;
}

// 电机初始化
void motor_init() {
    pi = pigpio_start(NULL, NULL);
    if (pi < 0) {
        std::cerr << "pigpio 连接失败！" << std::endl;
        exit(1);
    }

    set_mode(pi, LF_FWD, PI_OUTPUT);
    set_mode(pi, LF_BWD, PI_OUTPUT);
    set_mode(pi, LB_FWD, PI_OUTPUT);
    set_mode(pi, LB_BWD, PI_OUTPUT);
    set_mode(pi, RF_FWD, PI_OUTPUT);
    set_mode(pi, RF_BWD, PI_OUTPUT);
    set_mode(pi, RB_FWD, PI_OUTPUT);
    set_mode(pi, RB_BWD, PI_OUTPUT);

    set_mode(pi, LF_PWM, PI_OUTPUT);
    set_mode(pi, LB_PWM, PI_OUTPUT);
    set_mode(pi, RF_PWM, PI_OUTPUT);
    set_mode(pi, RB_PWM, PI_OUTPUT);
}

// 全部停止
void all_stop() {
    gpio_write(pi, LF_FWD, 0); gpio_write(pi, LF_BWD, 0);
    gpio_write(pi, LB_FWD, 0); gpio_write(pi, LB_BWD, 0);
    gpio_write(pi, RF_FWD, 0); gpio_write(pi, RF_BWD, 0);
    gpio_write(pi, RB_FWD, 0); gpio_write(pi, RB_BWD, 0);

    hardware_PWM(pi, LF_PWM, 0, 0);
    hardware_PWM(pi, LB_PWM, 0, 0);
    hardware_PWM(pi, RF_PWM, 0, 0);
    hardware_PWM(pi, RB_PWM, 0, 0);
}

// 单独测试一个电机：正转 → 停 → 反转
void test_motor(int fwd, int bwd, int pwm, const char* name) {
    std::cout << "→ 测试 " << name << " 正转" << std::endl;
    gpio_write(pi, fwd, 1);
    gpio_write(pi, bwd, 0);
    hardware_PWM(pi, pwm, PWM_FREQ, TEST_DUTY);
    sleep(1);

    all_stop();
    std::cout << "→ 停止" << std::endl;
    usleep(500000);

    std::cout << "→ 测试 " << name << " 反转" << std::endl;
    gpio_write(pi, fwd, 0);
    gpio_write(pi, bwd, 1);
    hardware_PWM(pi, pwm, PWM_FREQ, TEST_DUTY);
    sleep(1);

    all_stop();
    std::cout << "→ " << name << " 测试完成！\n" << std::endl;
}

int main() {
    motor_init();
    all_stop();

    std::cout << "===== 电机单独测试程序 =====" << std::endl;
    std::cout << "1: 左前   2: 左后   3: 右前   4: 右后   Q: 退出" << std::endl;

    while (1) {
        int c = kbhit();
        if (c == '1') test_motor(LF_FWD, LF_BWD, LF_PWM, "左前轮");
        if (c == '2') test_motor(LB_FWD, LB_BWD, LB_PWM, "左后轮");
        if (c == '3') test_motor(RF_FWD, RF_BWD, RF_PWM, "右前轮");
        if (c == '4') test_motor(RB_FWD, RB_BWD, RB_PWM, "右后轮");
        if (c == 'q' || c == 'Q') {
            std::cout << "退出程序" << std::endl;
            break;
        }
    }

    all_stop();
    pigpio_stop(pi);
    return 0;
}