int sum_to(int n) {
    int total = 0;
    for (int i = 1; i <= n; ++i) {
        total += i;
    }
    return total;
}

int main(int argc, char **argv) {
    return sum_to(argc) == 15 ? 0 : 1;
}
