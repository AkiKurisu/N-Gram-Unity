using Unity.Burst;
using Unity.Collections;
using Unity.Collections.LowLevel.Unsafe;
using Unity.Jobs;
namespace Kurisu.NGram
{
    /// <summary>
    /// Multi-thread 2~8 Gram implement based on -127~127 byte index using Job System
    /// Very lit, box 8 bytes into 1 double 
    /// 2 gram : use 1 inference 1
    /// 8 gram : use 7 inference 1
    /// </summary>
    [BurstCompile]
    public struct NGram8Job : IJob
    {
        #region Job ReadOnly Properties
        [ReadOnly]
        public NativeArray<byte> History;
        [ReadOnly]
        public int NGram;
        [ReadOnly]
        public NativeArray<byte> Inference;
        #endregion
        public NativeArray<double> Result;
        [BurstCompile]
        public void Execute()
        {
            int count = History.Length - NGram + 1;
#if UNITY_COLLECTIONS_1_3
            var predictions = new NativeParallelHashMap<double, double>(count, Allocator.Temp);
            //Dictionary<Pattern,OccurrenceKey>
            var worldOccurrence = new NativeParallelMultiHashMap<double, double>(count, Allocator.Temp);
            //Dictionary<OccurrenceKey,OccurrenceCount>
            var occurrenceMap = new NativeParallelHashMap<double, int>(count, Allocator.Temp);
#else
            var predictions = new NativeHashMap<double, double>(count, Allocator.Temp);
            var worldOccurrence = new NativeMultiHashMap<double, double>(count, Allocator.Temp);
            var occurrenceMap = new NativeHashMap<double, int>(count, Allocator.Temp);
#endif
            Compression(worldOccurrence, occurrenceMap);
            BuildPrediction(worldOccurrence, occurrenceMap, predictions);
            TryInference(predictions);
            predictions.Dispose();
            worldOccurrence.Dispose();
            occurrenceMap.Dispose();
        }
        //Compress byte[] to double (8 bytes)
        [BurstCompile]
#if UNITY_COLLECTIONS_1_3
        private void Compression(
            NativeParallelMultiHashMap<double, double> worldOccurrence,
            NativeParallelHashMap<double, int> occurrenceMap
         )
#else
        private void Compression(
            NativeMultiHashMap<double, double> worldOccurrence,
            NativeHashMap<double, int> occurrenceMap
         )
#endif
        {
            var buffer = new NativeArray<byte>(8, Allocator.Temp);
            int count = History.Length - NGram + 1;
            for (int i = 0; i < count; i++)
            {
                for (int j = 0; j < NGram - 1; ++j)
                {
                    buffer[j] = History[i + j];
                }
                //Ensure last one is zero
                for (int j = NGram - 1; j < 8; ++j)
                {
                    buffer[j] = 0;
                }
                //0~6 as key
                double key = buffer.Reinterpret<double>(UnsafeUtility.SizeOf<byte>())[0];
                //7 as predict
                buffer[NGram - 1] = History[i + NGram - 1];
                double occurrenceKey = buffer.Reinterpret<double>(UnsafeUtility.SizeOf<byte>())[0];
                if (occurrenceMap.TryGetValue(occurrenceKey, out int occurrences))
                {
                    occurrenceMap[occurrenceKey] = occurrences + 1;
                }
                else
                {
                    occurrenceMap[occurrenceKey] = 1;
                    worldOccurrence.Add(key, occurrenceKey);
                }
            }
            buffer.Dispose();
        }
        [BurstCompile]
#if UNITY_COLLECTIONS_1_3
        private void TryInference(NativeParallelHashMap<double, double> predictions)
#else
        private void TryInference(NativeHashMap<double, double> predictions)
#endif
        {
            var buffer = new NativeArray<byte>(8, Allocator.Temp);
            for (int j = 0; j < NGram - 1; ++j)
            {
                buffer[j] = Inference[^(NGram - 1 - j)];
            }
            for (int j = NGram - 1; j < 8; ++j)
            {
                buffer[j] = 0;
            }
            double key = buffer.Reinterpret<double>(UnsafeUtility.SizeOf<byte>())[0];
            buffer.Dispose();
            if (predictions.TryGetValue(key, out double result))
            {
                Result[0] = result;
            }
            else
            {
                Result[0] = -1;
            }
        }
        [BurstCompile]
#if UNITY_COLLECTIONS_1_3
        private readonly void BuildPrediction(
            NativeParallelMultiHashMap<double, double> worldOccurrence,
            NativeParallelHashMap<double, int> occurrenceMap,
            NativeParallelHashMap<double, double> predictions
        )
#else
        private readonly void BuildPrediction(
            NativeMultiHashMap<double, double> worldOccurrence,
            NativeHashMap<double, int> occurrenceMap,
            NativeHashMap<double, double> predictions
        )
#endif
        {
            var keys = worldOccurrence.GetKeyArray(Allocator.Temp);
            foreach (var start in keys)
            {
                double prediction = -1;
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