#include <filesystem>

void MyApplication();

int main() {
    std::filesystem::current_path("OpenCVExample");
    MyApplication();
}
