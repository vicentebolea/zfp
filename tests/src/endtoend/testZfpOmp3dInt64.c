#include "src/encode3l.c"

#include "constants/3dInt64.h"
#include "utils/hash64.h"
#include "ompExecBase.c"

int main()
{
  const struct CMUnitTest tests[] = {
    cmocka_unit_test(when_seededRandomSmoothDataGenerated_expect_ChecksumMatches),

    /* fixed-precision */
    cmocka_unit_test_setup_teardown(given_OpenMP_3dInt64Array_when_ZfpCompressFixedPrecision_expect_BitstreamChecksumMatches, setupFixedPrec0Param0Thread0Chunk, teardown),
    cmocka_unit_test_setup_teardown(given_OpenMP_3dInt64Array_when_ZfpDecompressFixedPrecision_expect_ArrayChecksumMatches, setupFixedPrec0Param0Thread0Chunk, teardown),
    cmocka_unit_test_setup_teardown(given_OpenMP_3dInt64Array_when_ZfpCompressFixedPrecision_expect_BitstreamChecksumMatches, setupFixedPrec0Param0Thread1Chunk, teardown),
    cmocka_unit_test_setup_teardown(given_OpenMP_3dInt64Array_when_ZfpDecompressFixedPrecision_expect_ArrayChecksumMatches, setupFixedPrec0Param0Thread1Chunk, teardown),
    cmocka_unit_test_setup_teardown(given_OpenMP_3dInt64Array_when_ZfpCompressFixedPrecision_expect_BitstreamChecksumMatches, setupFixedPrec0Param0Thread2Chunk, teardown),
    cmocka_unit_test_setup_teardown(given_OpenMP_3dInt64Array_when_ZfpDecompressFixedPrecision_expect_ArrayChecksumMatches, setupFixedPrec0Param0Thread2Chunk, teardown),

    cmocka_unit_test_setup_teardown(given_OpenMP_3dInt64Array_when_ZfpCompressFixedPrecision_expect_BitstreamChecksumMatches, setupFixedPrec0Param1Thread0Chunk, teardown),
    cmocka_unit_test_setup_teardown(given_OpenMP_3dInt64Array_when_ZfpDecompressFixedPrecision_expect_ArrayChecksumMatches, setupFixedPrec0Param1Thread0Chunk, teardown),
    cmocka_unit_test_setup_teardown(given_OpenMP_3dInt64Array_when_ZfpCompressFixedPrecision_expect_BitstreamChecksumMatches, setupFixedPrec0Param1Thread1Chunk, teardown),
    cmocka_unit_test_setup_teardown(given_OpenMP_3dInt64Array_when_ZfpDecompressFixedPrecision_expect_ArrayChecksumMatches, setupFixedPrec0Param1Thread1Chunk, teardown),
    cmocka_unit_test_setup_teardown(given_OpenMP_3dInt64Array_when_ZfpCompressFixedPrecision_expect_BitstreamChecksumMatches, setupFixedPrec0Param1Thread2Chunk, teardown),
    cmocka_unit_test_setup_teardown(given_OpenMP_3dInt64Array_when_ZfpDecompressFixedPrecision_expect_ArrayChecksumMatches, setupFixedPrec0Param1Thread2Chunk, teardown),

    cmocka_unit_test_setup_teardown(given_OpenMP_3dInt64Array_when_ZfpCompressFixedPrecision_expect_BitstreamChecksumMatches, setupFixedPrec0Param2Thread0Chunk, teardown),
    cmocka_unit_test_setup_teardown(given_OpenMP_3dInt64Array_when_ZfpDecompressFixedPrecision_expect_ArrayChecksumMatches, setupFixedPrec0Param2Thread0Chunk, teardown),
    cmocka_unit_test_setup_teardown(given_OpenMP_3dInt64Array_when_ZfpCompressFixedPrecision_expect_BitstreamChecksumMatches, setupFixedPrec0Param2Thread1Chunk, teardown),
    cmocka_unit_test_setup_teardown(given_OpenMP_3dInt64Array_when_ZfpDecompressFixedPrecision_expect_ArrayChecksumMatches, setupFixedPrec0Param2Thread1Chunk, teardown),
    cmocka_unit_test_setup_teardown(given_OpenMP_3dInt64Array_when_ZfpCompressFixedPrecision_expect_BitstreamChecksumMatches, setupFixedPrec0Param2Thread2Chunk, teardown),
    cmocka_unit_test_setup_teardown(given_OpenMP_3dInt64Array_when_ZfpDecompressFixedPrecision_expect_ArrayChecksumMatches, setupFixedPrec0Param2Thread2Chunk, teardown),

    cmocka_unit_test_setup_teardown(given_OpenMP_3dInt64Array_when_ZfpCompressFixedPrecision_expect_BitstreamChecksumMatches, setupFixedPrec1Param0Thread0Chunk, teardown),
    cmocka_unit_test_setup_teardown(given_OpenMP_3dInt64Array_when_ZfpDecompressFixedPrecision_expect_ArrayChecksumMatches, setupFixedPrec1Param0Thread0Chunk, teardown),
    cmocka_unit_test_setup_teardown(given_OpenMP_3dInt64Array_when_ZfpCompressFixedPrecision_expect_BitstreamChecksumMatches, setupFixedPrec1Param0Thread1Chunk, teardown),
    cmocka_unit_test_setup_teardown(given_OpenMP_3dInt64Array_when_ZfpDecompressFixedPrecision_expect_ArrayChecksumMatches, setupFixedPrec1Param0Thread1Chunk, teardown),
    cmocka_unit_test_setup_teardown(given_OpenMP_3dInt64Array_when_ZfpCompressFixedPrecision_expect_BitstreamChecksumMatches, setupFixedPrec1Param0Thread2Chunk, teardown),
    cmocka_unit_test_setup_teardown(given_OpenMP_3dInt64Array_when_ZfpDecompressFixedPrecision_expect_ArrayChecksumMatches, setupFixedPrec1Param0Thread2Chunk, teardown),

    cmocka_unit_test_setup_teardown(given_OpenMP_3dInt64Array_when_ZfpCompressFixedPrecision_expect_BitstreamChecksumMatches, setupFixedPrec1Param1Thread0Chunk, teardown),
    cmocka_unit_test_setup_teardown(given_OpenMP_3dInt64Array_when_ZfpDecompressFixedPrecision_expect_ArrayChecksumMatches, setupFixedPrec1Param1Thread0Chunk, teardown),
    cmocka_unit_test_setup_teardown(given_OpenMP_3dInt64Array_when_ZfpCompressFixedPrecision_expect_BitstreamChecksumMatches, setupFixedPrec1Param1Thread1Chunk, teardown),
    cmocka_unit_test_setup_teardown(given_OpenMP_3dInt64Array_when_ZfpDecompressFixedPrecision_expect_ArrayChecksumMatches, setupFixedPrec1Param1Thread1Chunk, teardown),
    cmocka_unit_test_setup_teardown(given_OpenMP_3dInt64Array_when_ZfpCompressFixedPrecision_expect_BitstreamChecksumMatches, setupFixedPrec1Param1Thread2Chunk, teardown),
    cmocka_unit_test_setup_teardown(given_OpenMP_3dInt64Array_when_ZfpDecompressFixedPrecision_expect_ArrayChecksumMatches, setupFixedPrec1Param1Thread2Chunk, teardown),

    cmocka_unit_test_setup_teardown(given_OpenMP_3dInt64Array_when_ZfpCompressFixedPrecision_expect_BitstreamChecksumMatches, setupFixedPrec1Param2Thread0Chunk, teardown),
    cmocka_unit_test_setup_teardown(given_OpenMP_3dInt64Array_when_ZfpDecompressFixedPrecision_expect_ArrayChecksumMatches, setupFixedPrec1Param2Thread0Chunk, teardown),
    cmocka_unit_test_setup_teardown(given_OpenMP_3dInt64Array_when_ZfpCompressFixedPrecision_expect_BitstreamChecksumMatches, setupFixedPrec1Param2Thread1Chunk, teardown),
    cmocka_unit_test_setup_teardown(given_OpenMP_3dInt64Array_when_ZfpDecompressFixedPrecision_expect_ArrayChecksumMatches, setupFixedPrec1Param2Thread1Chunk, teardown),
    cmocka_unit_test_setup_teardown(given_OpenMP_3dInt64Array_when_ZfpCompressFixedPrecision_expect_BitstreamChecksumMatches, setupFixedPrec1Param2Thread2Chunk, teardown),
    cmocka_unit_test_setup_teardown(given_OpenMP_3dInt64Array_when_ZfpDecompressFixedPrecision_expect_ArrayChecksumMatches, setupFixedPrec1Param2Thread2Chunk, teardown),

    cmocka_unit_test_setup_teardown(given_OpenMP_3dInt64Array_when_ZfpCompressFixedPrecision_expect_BitstreamChecksumMatches, setupFixedPrec2Param0Thread0Chunk, teardown),
    cmocka_unit_test_setup_teardown(given_OpenMP_3dInt64Array_when_ZfpDecompressFixedPrecision_expect_ArrayChecksumMatches, setupFixedPrec2Param0Thread0Chunk, teardown),
    cmocka_unit_test_setup_teardown(given_OpenMP_3dInt64Array_when_ZfpCompressFixedPrecision_expect_BitstreamChecksumMatches, setupFixedPrec2Param0Thread1Chunk, teardown),
    cmocka_unit_test_setup_teardown(given_OpenMP_3dInt64Array_when_ZfpDecompressFixedPrecision_expect_ArrayChecksumMatches, setupFixedPrec2Param0Thread1Chunk, teardown),
    cmocka_unit_test_setup_teardown(given_OpenMP_3dInt64Array_when_ZfpCompressFixedPrecision_expect_BitstreamChecksumMatches, setupFixedPrec2Param0Thread2Chunk, teardown),
    cmocka_unit_test_setup_teardown(given_OpenMP_3dInt64Array_when_ZfpDecompressFixedPrecision_expect_ArrayChecksumMatches, setupFixedPrec2Param0Thread2Chunk, teardown),

    cmocka_unit_test_setup_teardown(given_OpenMP_3dInt64Array_when_ZfpCompressFixedPrecision_expect_BitstreamChecksumMatches, setupFixedPrec2Param1Thread0Chunk, teardown),
    cmocka_unit_test_setup_teardown(given_OpenMP_3dInt64Array_when_ZfpDecompressFixedPrecision_expect_ArrayChecksumMatches, setupFixedPrec2Param1Thread0Chunk, teardown),
    cmocka_unit_test_setup_teardown(given_OpenMP_3dInt64Array_when_ZfpCompressFixedPrecision_expect_BitstreamChecksumMatches, setupFixedPrec2Param1Thread1Chunk, teardown),
    cmocka_unit_test_setup_teardown(given_OpenMP_3dInt64Array_when_ZfpDecompressFixedPrecision_expect_ArrayChecksumMatches, setupFixedPrec2Param1Thread1Chunk, teardown),
    cmocka_unit_test_setup_teardown(given_OpenMP_3dInt64Array_when_ZfpCompressFixedPrecision_expect_BitstreamChecksumMatches, setupFixedPrec2Param1Thread2Chunk, teardown),
    cmocka_unit_test_setup_teardown(given_OpenMP_3dInt64Array_when_ZfpDecompressFixedPrecision_expect_ArrayChecksumMatches, setupFixedPrec2Param1Thread2Chunk, teardown),

    cmocka_unit_test_setup_teardown(given_OpenMP_3dInt64Array_when_ZfpCompressFixedPrecision_expect_BitstreamChecksumMatches, setupFixedPrec2Param2Thread0Chunk, teardown),
    cmocka_unit_test_setup_teardown(given_OpenMP_3dInt64Array_when_ZfpDecompressFixedPrecision_expect_ArrayChecksumMatches, setupFixedPrec2Param2Thread0Chunk, teardown),
    cmocka_unit_test_setup_teardown(given_OpenMP_3dInt64Array_when_ZfpCompressFixedPrecision_expect_BitstreamChecksumMatches, setupFixedPrec2Param2Thread1Chunk, teardown),
    cmocka_unit_test_setup_teardown(given_OpenMP_3dInt64Array_when_ZfpDecompressFixedPrecision_expect_ArrayChecksumMatches, setupFixedPrec2Param2Thread1Chunk, teardown),
    cmocka_unit_test_setup_teardown(given_OpenMP_3dInt64Array_when_ZfpCompressFixedPrecision_expect_BitstreamChecksumMatches, setupFixedPrec2Param2Thread2Chunk, teardown),
    cmocka_unit_test_setup_teardown(given_OpenMP_3dInt64Array_when_ZfpDecompressFixedPrecision_expect_ArrayChecksumMatches, setupFixedPrec2Param2Thread2Chunk, teardown),

    /* fixed-rate */
    cmocka_unit_test_setup_teardown(given_OpenMP_3dInt64Array_when_ZfpCompressFixedRate_expect_BitstreamChecksumMatches, setupFixedRate0Param0Thread0Chunk, teardown),
    cmocka_unit_test_setup_teardown(given_OpenMP_3dInt64Array_when_ZfpDecompressFixedRate_expect_ArrayChecksumMatches, setupFixedRate0Param0Thread0Chunk, teardown),
    cmocka_unit_test_setup_teardown(given_OpenMP_3dInt64Array_when_ZfpCompressFixedRate_expect_CompressedBitrateComparableToChosenRate, setupFixedRate0Param0Thread0Chunk, teardown),
    cmocka_unit_test_setup_teardown(given_OpenMP_3dInt64Array_when_ZfpCompressFixedRate_expect_BitstreamChecksumMatches, setupFixedRate0Param0Thread1Chunk, teardown),
    cmocka_unit_test_setup_teardown(given_OpenMP_3dInt64Array_when_ZfpDecompressFixedRate_expect_ArrayChecksumMatches, setupFixedRate0Param0Thread1Chunk, teardown),
    cmocka_unit_test_setup_teardown(given_OpenMP_3dInt64Array_when_ZfpCompressFixedRate_expect_CompressedBitrateComparableToChosenRate, setupFixedRate0Param0Thread1Chunk, teardown),
    cmocka_unit_test_setup_teardown(given_OpenMP_3dInt64Array_when_ZfpCompressFixedRate_expect_BitstreamChecksumMatches, setupFixedRate0Param0Thread2Chunk, teardown),
    cmocka_unit_test_setup_teardown(given_OpenMP_3dInt64Array_when_ZfpDecompressFixedRate_expect_ArrayChecksumMatches, setupFixedRate0Param0Thread2Chunk, teardown),
    cmocka_unit_test_setup_teardown(given_OpenMP_3dInt64Array_when_ZfpCompressFixedRate_expect_CompressedBitrateComparableToChosenRate, setupFixedRate0Param0Thread2Chunk, teardown),

    cmocka_unit_test_setup_teardown(given_OpenMP_3dInt64Array_when_ZfpCompressFixedRate_expect_BitstreamChecksumMatches, setupFixedRate0Param1Thread0Chunk, teardown),
    cmocka_unit_test_setup_teardown(given_OpenMP_3dInt64Array_when_ZfpDecompressFixedRate_expect_ArrayChecksumMatches, setupFixedRate0Param1Thread0Chunk, teardown),
    cmocka_unit_test_setup_teardown(given_OpenMP_3dInt64Array_when_ZfpCompressFixedRate_expect_CompressedBitrateComparableToChosenRate, setupFixedRate0Param1Thread0Chunk, teardown),
    cmocka_unit_test_setup_teardown(given_OpenMP_3dInt64Array_when_ZfpCompressFixedRate_expect_BitstreamChecksumMatches, setupFixedRate0Param1Thread1Chunk, teardown),
    cmocka_unit_test_setup_teardown(given_OpenMP_3dInt64Array_when_ZfpDecompressFixedRate_expect_ArrayChecksumMatches, setupFixedRate0Param1Thread1Chunk, teardown),
    cmocka_unit_test_setup_teardown(given_OpenMP_3dInt64Array_when_ZfpCompressFixedRate_expect_CompressedBitrateComparableToChosenRate, setupFixedRate0Param1Thread1Chunk, teardown),
    cmocka_unit_test_setup_teardown(given_OpenMP_3dInt64Array_when_ZfpCompressFixedRate_expect_BitstreamChecksumMatches, setupFixedRate0Param1Thread2Chunk, teardown),
    cmocka_unit_test_setup_teardown(given_OpenMP_3dInt64Array_when_ZfpDecompressFixedRate_expect_ArrayChecksumMatches, setupFixedRate0Param1Thread2Chunk, teardown),
    cmocka_unit_test_setup_teardown(given_OpenMP_3dInt64Array_when_ZfpCompressFixedRate_expect_CompressedBitrateComparableToChosenRate, setupFixedRate0Param1Thread2Chunk, teardown),

    cmocka_unit_test_setup_teardown(given_OpenMP_3dInt64Array_when_ZfpCompressFixedRate_expect_BitstreamChecksumMatches, setupFixedRate0Param2Thread0Chunk, teardown),
    cmocka_unit_test_setup_teardown(given_OpenMP_3dInt64Array_when_ZfpDecompressFixedRate_expect_ArrayChecksumMatches, setupFixedRate0Param2Thread0Chunk, teardown),
    cmocka_unit_test_setup_teardown(given_OpenMP_3dInt64Array_when_ZfpCompressFixedRate_expect_CompressedBitrateComparableToChosenRate, setupFixedRate0Param2Thread0Chunk, teardown),
    cmocka_unit_test_setup_teardown(given_OpenMP_3dInt64Array_when_ZfpCompressFixedRate_expect_BitstreamChecksumMatches, setupFixedRate0Param2Thread1Chunk, teardown),
    cmocka_unit_test_setup_teardown(given_OpenMP_3dInt64Array_when_ZfpDecompressFixedRate_expect_ArrayChecksumMatches, setupFixedRate0Param2Thread1Chunk, teardown),
    cmocka_unit_test_setup_teardown(given_OpenMP_3dInt64Array_when_ZfpCompressFixedRate_expect_CompressedBitrateComparableToChosenRate, setupFixedRate0Param2Thread1Chunk, teardown),
    cmocka_unit_test_setup_teardown(given_OpenMP_3dInt64Array_when_ZfpCompressFixedRate_expect_BitstreamChecksumMatches, setupFixedRate0Param2Thread2Chunk, teardown),
    cmocka_unit_test_setup_teardown(given_OpenMP_3dInt64Array_when_ZfpDecompressFixedRate_expect_ArrayChecksumMatches, setupFixedRate0Param2Thread2Chunk, teardown),
    cmocka_unit_test_setup_teardown(given_OpenMP_3dInt64Array_when_ZfpCompressFixedRate_expect_CompressedBitrateComparableToChosenRate, setupFixedRate0Param2Thread2Chunk, teardown),

    cmocka_unit_test_setup_teardown(given_OpenMP_3dInt64Array_when_ZfpCompressFixedRate_expect_BitstreamChecksumMatches, setupFixedRate1Param0Thread0Chunk, teardown),
    cmocka_unit_test_setup_teardown(given_OpenMP_3dInt64Array_when_ZfpDecompressFixedRate_expect_ArrayChecksumMatches, setupFixedRate1Param0Thread0Chunk, teardown),
    cmocka_unit_test_setup_teardown(given_OpenMP_3dInt64Array_when_ZfpCompressFixedRate_expect_CompressedBitrateComparableToChosenRate, setupFixedRate1Param0Thread0Chunk, teardown),
    cmocka_unit_test_setup_teardown(given_OpenMP_3dInt64Array_when_ZfpCompressFixedRate_expect_BitstreamChecksumMatches, setupFixedRate1Param0Thread1Chunk, teardown),
    cmocka_unit_test_setup_teardown(given_OpenMP_3dInt64Array_when_ZfpDecompressFixedRate_expect_ArrayChecksumMatches, setupFixedRate1Param0Thread1Chunk, teardown),
    cmocka_unit_test_setup_teardown(given_OpenMP_3dInt64Array_when_ZfpCompressFixedRate_expect_CompressedBitrateComparableToChosenRate, setupFixedRate1Param0Thread1Chunk, teardown),
    cmocka_unit_test_setup_teardown(given_OpenMP_3dInt64Array_when_ZfpCompressFixedRate_expect_BitstreamChecksumMatches, setupFixedRate1Param0Thread2Chunk, teardown),
    cmocka_unit_test_setup_teardown(given_OpenMP_3dInt64Array_when_ZfpDecompressFixedRate_expect_ArrayChecksumMatches, setupFixedRate1Param0Thread2Chunk, teardown),
    cmocka_unit_test_setup_teardown(given_OpenMP_3dInt64Array_when_ZfpCompressFixedRate_expect_CompressedBitrateComparableToChosenRate, setupFixedRate1Param0Thread2Chunk, teardown),

    cmocka_unit_test_setup_teardown(given_OpenMP_3dInt64Array_when_ZfpCompressFixedRate_expect_BitstreamChecksumMatches, setupFixedRate1Param1Thread0Chunk, teardown),
    cmocka_unit_test_setup_teardown(given_OpenMP_3dInt64Array_when_ZfpDecompressFixedRate_expect_ArrayChecksumMatches, setupFixedRate1Param1Thread0Chunk, teardown),
    cmocka_unit_test_setup_teardown(given_OpenMP_3dInt64Array_when_ZfpCompressFixedRate_expect_CompressedBitrateComparableToChosenRate, setupFixedRate1Param1Thread0Chunk, teardown),
    cmocka_unit_test_setup_teardown(given_OpenMP_3dInt64Array_when_ZfpCompressFixedRate_expect_BitstreamChecksumMatches, setupFixedRate1Param1Thread1Chunk, teardown),
    cmocka_unit_test_setup_teardown(given_OpenMP_3dInt64Array_when_ZfpDecompressFixedRate_expect_ArrayChecksumMatches, setupFixedRate1Param1Thread1Chunk, teardown),
    cmocka_unit_test_setup_teardown(given_OpenMP_3dInt64Array_when_ZfpCompressFixedRate_expect_CompressedBitrateComparableToChosenRate, setupFixedRate1Param1Thread1Chunk, teardown),
    cmocka_unit_test_setup_teardown(given_OpenMP_3dInt64Array_when_ZfpCompressFixedRate_expect_BitstreamChecksumMatches, setupFixedRate1Param1Thread2Chunk, teardown),
    cmocka_unit_test_setup_teardown(given_OpenMP_3dInt64Array_when_ZfpDecompressFixedRate_expect_ArrayChecksumMatches, setupFixedRate1Param1Thread2Chunk, teardown),
    cmocka_unit_test_setup_teardown(given_OpenMP_3dInt64Array_when_ZfpCompressFixedRate_expect_CompressedBitrateComparableToChosenRate, setupFixedRate1Param1Thread2Chunk, teardown),

    cmocka_unit_test_setup_teardown(given_OpenMP_3dInt64Array_when_ZfpCompressFixedRate_expect_BitstreamChecksumMatches, setupFixedRate1Param2Thread0Chunk, teardown),
    cmocka_unit_test_setup_teardown(given_OpenMP_3dInt64Array_when_ZfpDecompressFixedRate_expect_ArrayChecksumMatches, setupFixedRate1Param2Thread0Chunk, teardown),
    cmocka_unit_test_setup_teardown(given_OpenMP_3dInt64Array_when_ZfpCompressFixedRate_expect_CompressedBitrateComparableToChosenRate, setupFixedRate1Param2Thread0Chunk, teardown),
    cmocka_unit_test_setup_teardown(given_OpenMP_3dInt64Array_when_ZfpCompressFixedRate_expect_BitstreamChecksumMatches, setupFixedRate1Param2Thread1Chunk, teardown),
    cmocka_unit_test_setup_teardown(given_OpenMP_3dInt64Array_when_ZfpDecompressFixedRate_expect_ArrayChecksumMatches, setupFixedRate1Param2Thread1Chunk, teardown),
    cmocka_unit_test_setup_teardown(given_OpenMP_3dInt64Array_when_ZfpCompressFixedRate_expect_CompressedBitrateComparableToChosenRate, setupFixedRate1Param2Thread1Chunk, teardown),
    cmocka_unit_test_setup_teardown(given_OpenMP_3dInt64Array_when_ZfpCompressFixedRate_expect_BitstreamChecksumMatches, setupFixedRate1Param2Thread2Chunk, teardown),
    cmocka_unit_test_setup_teardown(given_OpenMP_3dInt64Array_when_ZfpDecompressFixedRate_expect_ArrayChecksumMatches, setupFixedRate1Param2Thread2Chunk, teardown),
    cmocka_unit_test_setup_teardown(given_OpenMP_3dInt64Array_when_ZfpCompressFixedRate_expect_CompressedBitrateComparableToChosenRate, setupFixedRate1Param2Thread2Chunk, teardown),

    cmocka_unit_test_setup_teardown(given_OpenMP_3dInt64Array_when_ZfpCompressFixedRate_expect_BitstreamChecksumMatches, setupFixedRate2Param0Thread0Chunk, teardown),
    cmocka_unit_test_setup_teardown(given_OpenMP_3dInt64Array_when_ZfpDecompressFixedRate_expect_ArrayChecksumMatches, setupFixedRate2Param0Thread0Chunk, teardown),
    cmocka_unit_test_setup_teardown(given_OpenMP_3dInt64Array_when_ZfpCompressFixedRate_expect_CompressedBitrateComparableToChosenRate, setupFixedRate2Param0Thread0Chunk, teardown),
    cmocka_unit_test_setup_teardown(given_OpenMP_3dInt64Array_when_ZfpCompressFixedRate_expect_BitstreamChecksumMatches, setupFixedRate2Param0Thread1Chunk, teardown),
    cmocka_unit_test_setup_teardown(given_OpenMP_3dInt64Array_when_ZfpDecompressFixedRate_expect_ArrayChecksumMatches, setupFixedRate2Param0Thread1Chunk, teardown),
    cmocka_unit_test_setup_teardown(given_OpenMP_3dInt64Array_when_ZfpCompressFixedRate_expect_CompressedBitrateComparableToChosenRate, setupFixedRate2Param0Thread1Chunk, teardown),
    cmocka_unit_test_setup_teardown(given_OpenMP_3dInt64Array_when_ZfpCompressFixedRate_expect_BitstreamChecksumMatches, setupFixedRate2Param0Thread2Chunk, teardown),
    cmocka_unit_test_setup_teardown(given_OpenMP_3dInt64Array_when_ZfpDecompressFixedRate_expect_ArrayChecksumMatches, setupFixedRate2Param0Thread2Chunk, teardown),
    cmocka_unit_test_setup_teardown(given_OpenMP_3dInt64Array_when_ZfpCompressFixedRate_expect_CompressedBitrateComparableToChosenRate, setupFixedRate2Param0Thread2Chunk, teardown),

    cmocka_unit_test_setup_teardown(given_OpenMP_3dInt64Array_when_ZfpCompressFixedRate_expect_BitstreamChecksumMatches, setupFixedRate2Param1Thread0Chunk, teardown),
    cmocka_unit_test_setup_teardown(given_OpenMP_3dInt64Array_when_ZfpDecompressFixedRate_expect_ArrayChecksumMatches, setupFixedRate2Param1Thread0Chunk, teardown),
    cmocka_unit_test_setup_teardown(given_OpenMP_3dInt64Array_when_ZfpCompressFixedRate_expect_CompressedBitrateComparableToChosenRate, setupFixedRate2Param1Thread0Chunk, teardown),
    cmocka_unit_test_setup_teardown(given_OpenMP_3dInt64Array_when_ZfpCompressFixedRate_expect_BitstreamChecksumMatches, setupFixedRate2Param1Thread1Chunk, teardown),
    cmocka_unit_test_setup_teardown(given_OpenMP_3dInt64Array_when_ZfpDecompressFixedRate_expect_ArrayChecksumMatches, setupFixedRate2Param1Thread1Chunk, teardown),
    cmocka_unit_test_setup_teardown(given_OpenMP_3dInt64Array_when_ZfpCompressFixedRate_expect_CompressedBitrateComparableToChosenRate, setupFixedRate2Param1Thread1Chunk, teardown),
    cmocka_unit_test_setup_teardown(given_OpenMP_3dInt64Array_when_ZfpCompressFixedRate_expect_BitstreamChecksumMatches, setupFixedRate2Param1Thread2Chunk, teardown),
    cmocka_unit_test_setup_teardown(given_OpenMP_3dInt64Array_when_ZfpDecompressFixedRate_expect_ArrayChecksumMatches, setupFixedRate2Param1Thread2Chunk, teardown),
    cmocka_unit_test_setup_teardown(given_OpenMP_3dInt64Array_when_ZfpCompressFixedRate_expect_CompressedBitrateComparableToChosenRate, setupFixedRate2Param1Thread2Chunk, teardown),

    cmocka_unit_test_setup_teardown(given_OpenMP_3dInt64Array_when_ZfpCompressFixedRate_expect_BitstreamChecksumMatches, setupFixedRate2Param2Thread0Chunk, teardown),
    cmocka_unit_test_setup_teardown(given_OpenMP_3dInt64Array_when_ZfpDecompressFixedRate_expect_ArrayChecksumMatches, setupFixedRate2Param2Thread0Chunk, teardown),
    cmocka_unit_test_setup_teardown(given_OpenMP_3dInt64Array_when_ZfpCompressFixedRate_expect_CompressedBitrateComparableToChosenRate, setupFixedRate2Param2Thread0Chunk, teardown),
    cmocka_unit_test_setup_teardown(given_OpenMP_3dInt64Array_when_ZfpCompressFixedRate_expect_BitstreamChecksumMatches, setupFixedRate2Param2Thread1Chunk, teardown),
    cmocka_unit_test_setup_teardown(given_OpenMP_3dInt64Array_when_ZfpDecompressFixedRate_expect_ArrayChecksumMatches, setupFixedRate2Param2Thread1Chunk, teardown),
    cmocka_unit_test_setup_teardown(given_OpenMP_3dInt64Array_when_ZfpCompressFixedRate_expect_CompressedBitrateComparableToChosenRate, setupFixedRate2Param2Thread1Chunk, teardown),
    cmocka_unit_test_setup_teardown(given_OpenMP_3dInt64Array_when_ZfpCompressFixedRate_expect_BitstreamChecksumMatches, setupFixedRate2Param2Thread2Chunk, teardown),
    cmocka_unit_test_setup_teardown(given_OpenMP_3dInt64Array_when_ZfpDecompressFixedRate_expect_ArrayChecksumMatches, setupFixedRate2Param2Thread2Chunk, teardown),
    cmocka_unit_test_setup_teardown(given_OpenMP_3dInt64Array_when_ZfpCompressFixedRate_expect_CompressedBitrateComparableToChosenRate, setupFixedRate2Param2Thread2Chunk, teardown),
  };

  return cmocka_run_group_tests(tests, setupRandomData, teardownRandomData);
}
