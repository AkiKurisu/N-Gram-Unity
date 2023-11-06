using Unity.Burst;
using Unity.Collections;
using Unity.Collections.LowLevel.Unsafe;
using Unity.Jobs;
namespace Kurisu.NGram
{

    /// <summary>
    /// Multi-thread N Gram almost without limited
    /// Use XXHash to index, may exist hash collision when input id range or n gram become larger
    /// </summary>
    [BurstCompile]
    public struct NGramParallelJob : IJobParallelFor
    {

        #region Job ReadOnly Properties
        [ReadOnly]
        public NativeArray<int> History;
        [ReadOnly]
        public int StartGram;
        [ReadOnly]
        public NativeArray<int> Inference;
        #endregion
        public NativeArray<int> Result;
        [BurstCompile]
        public void Execute(int index)
        {
            int NGram = StartGram + index;
            int count = History.Length - NGram + 1;
#if UNITY_COLLECTIONS_1_3
            var predictions = new NativeParallelHashMap<uint, uint>(count, Allocator.Temp);
            //Dictionary<Pattern,OccurrenceKey>
            var worldOccurrence = new NativeParallelMultiHashMap<uint, uint>(count, Allocator.Temp);
            //Dictionary<OccurrenceKey,OccurrenceCount>
            var occurrenceMap = new NativeParallelHashMap<uint, int>(count, Allocator.Temp);
            //Dictionary<OccurrenceKey,OccurrenceCount>
            var pointerMap = new NativeParallelHashMap<uint, int>(count, Allocator.Temp);
#else
            var predictions = new NativeHashMap<uint, uint>(count, Allocator.Temp);
            var worldOccurrence = new NativeMultiHashMap<uint, uint>(count, Allocator.Temp);
            var occurrenceMap = new NativeHashMap<uint, int>(count, Allocator.Temp);
            var pointerMap = new NativeHashMap<uint, int>(count, Allocator.Temp);
#endif
            Compression(worldOccurrence, occurrenceMap, pointerMap, NGram);
            BuildPrediction(worldOccurrence, occurrenceMap, predictions);
            TryInference(index, predictions, pointerMap, NGram);
            predictions.Dispose();
            worldOccurrence.Dispose();
            occurrenceMap.Dispose();
            pointerMap.Dispose();
        }
        [BurstCompile]
#if UNITY_COLLECTIONS_1_3
        private void Compression(
            NativeParallelMultiHashMap<uint, uint> worldOccurrence,
            NativeParallelHashMap<uint, int> occurrenceMap,
            NativeParallelHashMap<uint, int> pointerMap,
            int NGram
         )
#else
        private void Compression(
            NativeMultiHashMap<uint, uint> worldOccurrence,
            NativeHashMap<uint, int> occurrenceMap,
            NativeHashMap<uint, int> pointerMap,
            int NGram
         )
#endif
        {
            int count = History.Length - NGram + 1;
            for (int i = 0; i < count; i++)
            {
                uint key = GetHash(History, i, false, NGram);
                uint occurrenceKey = GetHash(History, i, true, NGram);
                if (occurrenceMap.TryGetValue(occurrenceKey, out int occurrences))
                {
                    occurrenceMap[occurrenceKey] = occurrences + 1;
                }
                else
                {
                    //Store hash => pointer/start index
                    pointerMap[key] = i;
                    pointerMap[occurrenceKey] = i;
                    occurrenceMap[occurrenceKey] = 1;
                    worldOccurrence.Add(key, occurrenceKey);
                }
            }
        }
        private readonly uint GetHash(NativeArray<int> array, int startIndex, bool includeEnd, int length)
        {
            var buffer = new NativeArray<int>(length, Allocator.Temp);
            for (int j = 0; j < length - 1; ++j)
            {
                buffer[j] = array[startIndex + j];
            }
            buffer[length - 1] = includeEnd ? array[startIndex + length - 1] : 0;
            var hash = XXHash.CalculateHash(buffer.Reinterpret<byte>(UnsafeUtility.SizeOf<int>()));
            return hash;
        }
        [BurstCompile]
#if UNITY_COLLECTIONS_1_3
        private void TryInference(int index, NativeParallelHashMap<uint, uint> predictions, NativeParallelHashMap<uint, int> pointerMap, int NGram)
#else
        private void TryInference(int index, NativeHashMap<uint, uint> predictions,NativeHashMap<uint, int> pointerMap, int NGram)
#endif
        {
            uint key = GetHash(Inference, Inference.Length - NGram + 1, false, NGram);
            if (predictions.TryGetValue(key, out uint result))
            {
                Result[index] = History[pointerMap[result] + NGram - 1];
            }
            else
            {
                Result[index] = -1;
            }
        }
        [BurstCompile]
#if UNITY_COLLECTIONS_1_3
        private readonly void BuildPrediction(
            NativeParallelMultiHashMap<uint, uint> worldOccurrence,
            NativeParallelHashMap<uint, int> occurrenceMap,
            NativeParallelHashMap<uint, uint> predictions
        )
#else
        private readonly void BuildPrediction(
            NativeMultiHashMap<uint, uint> worldOccurrence,
            NativeHashMap<uint, int> occurrenceMap,
            NativeHashMap<uint, uint> predictions
        )
#endif
        {
            var keys = worldOccurrence.GetKeyArray(Allocator.Temp);
            foreach (var start in keys)
            {
                uint prediction = 0;
                int maximum = int.MinValue;
                foreach (var end in worldOccurrence.GetValuesForKey(start))
                {
                    if (occurrenceMap[end] > maximum)
                    {
                        prediction = end;
                        maximum = occurrenceMap[end];
                    }
                }
                predictions[start] = prediction;
            }
            keys.Dispose();
        }
    }
}