/* tests for runtime/null/core.h: element counts and broadcast strides; included by test_kernels.c */

static void test_numel(void) {
    int64_t s[4] = {2, 3, 4, 5};
    int64_t z[2] = {3, 0};
    CHECK(null_numel(4, s) == 120);
    CHECK(null_numel(0, s) == 1);   /* scalar */
    CHECK(null_numel(2, z) == 0);   /* empty */
}

static void test_broadcast_strides(void) {
    int64_t out[2] = {2, 3};
    int64_t full[2] = {2, 3}, row[1] = {3}, col[2] = {2, 1}, one[1] = {1};
    int64_t st[2];

    null_broadcast_strides(2, out, 2, full, st);
    CHECK(st[0] == 3 && st[1] == 1);          /* contiguous */

    null_broadcast_strides(2, out, 1, row, st);
    CHECK(st[0] == 0 && st[1] == 1);          /* missing leading dim */

    null_broadcast_strides(2, out, 2, col, st);
    CHECK(st[0] == 1 && st[1] == 0);          /* size-1 trailing dim */

    null_broadcast_strides(2, out, 1, one, st);
    CHECK(st[0] == 0 && st[1] == 0);          /* broadcast scalar */

    null_broadcast_strides(2, out, 0, one, st);
    CHECK(st[0] == 0 && st[1] == 0);          /* rank 0 input */

    {
        /* [8, 1, 6, 1] read into [8, 7, 6, 5] */
        int64_t o[4] = {8, 7, 6, 5}, a[4] = {8, 1, 6, 1}, s4[4];
        null_broadcast_strides(4, o, 4, a, s4);
        CHECK(s4[0] == 6 && s4[1] == 0 && s4[2] == 1 && s4[3] == 0);
    }
}

static void run_core_tests(void) {
    test_numel();
    test_broadcast_strides();
}
