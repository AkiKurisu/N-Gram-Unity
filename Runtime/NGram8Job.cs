using Unity.Burst;
using Unity.Collections;
using Unity.Collections.LowLevel.Unsafe;
using Unity.Jobs;
namespace Kurisu.NGram
{
    /// <summary>
    /// Multi-thread 2~8 Gram implement using Job System
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
            var predictions = new NativeHashMap<double, double>(count, Allocator.Temp);
            //Dictionary<Pattern,OccurenceKey>
            var worldOccurence = new NativeMultiHashMap<double, double>(count, Allocator.Temp);
            //Dictionary<OccurenceKey,OccurenceCount>
            var occurenceMap = new NativeHashMap<double, int>(count, Allocator.Temp);
            Compression(worldOccurence, occurenceMap);
            BuildPrediction(worldOccurence, occurenceMap, predictions);
            TryInference(predictions);
            predictions.Dispose();
            worldOccurence.Dispose();
            occurenceMap.Dispose();
        }
        //Compress byte[] to double (8 bytes)
        [BurstCompile]
        private void Compression(
            NativeMultiHashMap<double, double> worldOccurence,
            NativeHashMap<double, int> occurenceMap
         )
        {
            var buffer = new NativeArray<byte>(8, Allocator.Temp);
            int count = History.Length - NGram + 1;
            for (int i = 0; i < count; i++)
            {
                for (int j = 0; j < 8; j++)
                {
                    buffer[j] = (j < NGram - 1) ? History[i + j] : (byte)0;
                }
                double key = buffer.Reinterpret<double>(UnsafeUtility.SizeOf<byte>())[0];
                buffer[7] = History[i + 7];
                int occurenceKey = buffer.Reinterpret<int>(UnsafeUtility.SizeOf<byte>())[0];
                if (occurenceMap.TryGetValue(occurenceKey, out int occurences))
                {
                    occurenceMap[occurenceKey] = occurences + 1;
                }
                else
                {
                    occurenceMap[occurenceKey] = 1;
                    worldOccurence.Add(key, occurenceKey);
                }
            }
            buffer.Dispose();
        }
        [BurstCompile]
        private void TryInference(NativeHashMap<double, double> predictions)
        {
            var buffer = new NativeArray<byte>(8, Allocator.Temp);
            for (int i = 0; i < 8; i++)
            {
                buffer[i] = (i < NGram - 1) ? Inference[^(NGram - 1 - i)] : (byte)0;
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
        private readonly void BuildPrediction(
             NativeMultiHashMap<double, double> worldOccurence,
            NativeHashMap<double, int> occurenceMap,
            NativeHashMap<double, double> predictions
        )
        {
            var keys = worldOccurence.GetKeyArray(Allocator.Temp);
            foreach (var start in keys)
            {
                double prediction = -1;
                double maximum = double.MinValue;
                foreach (var end in worldOccurence.GetValuesForKey(start))
                {
                    if (occurenceMap[end] > maximum)
                    {
                        prediction = end;
                        maximum = occurenceMap[end];
                    }
                }
                predictions[start] = prediction;
            }
            keys.Dispose();
        }
    }
}