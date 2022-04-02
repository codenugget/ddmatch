#include <matplot/matplot.h>
#include <string>
#include <thread>

int main(int argc, char** argv) {
    using namespace matplot;

    std::string filename = (argc > 1) ? std::string(argv[1]) : "";
    bool save_to_file = filename != "";
    vector_1d x = linspace(-2 * pi, 2 * pi);
    vector_1d y = linspace(0, 4 * pi);
    auto [X, Y] = meshgrid(x, y);
    if (save_to_file)
        gcf(true); // run in quiet mode
    vector_2d Z =
        transform(X, Y, [](double x, double y) { return sin(x) + cos(y); });
    contour(X, Y, Z);

    if (save_to_file)
        save(filename.c_str());
    else
        show();
    return 0;
}
