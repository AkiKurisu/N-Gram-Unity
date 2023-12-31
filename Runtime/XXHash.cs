using Unity.Burst;
using Unity.Collections;
using Unity.Collections.LowLevel.Unsafe;
namespace Kurisu.NGram
{
    public static class XXHash
    {
        private const uint PRIME32_1 = 2654435761U;
        private const uint PRIME32_2 = 2246822519U;
        private const uint PRIME32_3 = 3266489917U;
        private const uint PRIME32_4 = 668265263U;
        private const uint PRIME32_5 = 374761393U;
        [BurstCompile]
        public static uint CalculateHash(NativeArray<byte> data)
        {
            uint seed = 0;
            uint hash = seed + PRIME32_5;
            int currentIndex = 0;
            int remainingBytes = data.Length;
            while (remainingBytes >= 4)
            {
                uint currentUint = data.Reinterpret<uint>(UnsafeUtility.SizeOf<byte>())[currentIndex / 4];
                currentUint *= PRIME32_3;
                currentUint = RotateLeft(currentUint, 17) * PRIME32_4;
                hash ^= currentUint;
                hash = RotateLeft(hash, 19);
                hash = hash * PRIME32_1 + PRIME32_4;
                currentIndex += 4;
                remainingBytes -= 4;
            }

            while (remainingBytes > 0)
            {
                hash ^= data[currentIndex] * PRIME32_5;
                hash = RotateLeft(hash, 11) * PRIME32_1;
                currentIndex++;
                remainingBytes--;
            }

            hash ^= (uint)data.Length;
            hash ^= hash >> 15;
            hash *= PRIME32_2;
            hash ^= hash >> 13;
            hash *= PRIME32_3;
            hash ^= hash >> 16;

            return hash;
        }

        private static uint RotateLeft(uint value, int count)
        {
            return (value << count) | (value >> (32 - count));
        }
    }

}