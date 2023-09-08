using Unity.Burst;
using Unity.Collections;
using Unity.Collections.LowLevel.Unsafe;
using Unity.Jobs;
namespace Kurisu.NGram
{
    /// <summary>
    /// Multi-thread N-Gram implement using Job System
    /// </summary>
    [BurstCompile]
    public struct NGramJob : IJob
    {
        #region Job ReadOnly Property
        [ReadOnly]
        public NativeArray<byte> History;
        [ReadOnly]
        public int NGram;
        [ReadOnly]
        public int Inference;
        #endregion
        //Result
        public NativeArray<int> Result;
        [BurstCompile]
        public void Execute()
        {
            int count = History.Length - NGram + 1;
            var predictions = new NativeHashMap<int, int>(count, Allocator.Temp);
            //Dictionary<Pattern,OccurenceKey>
            var worldOccurence = new NativeMultiHashMap<int, int>(count, Allocator.Temp);
            //Dictionary<OccurenceKey,OccurenceCount>
            var occurenceMap = new NativeHashMap<int, int>(count, Allocator.Temp);
            Compression(worldOccurence, occurenceMap);
            BuildPrediction(worldOccurence, occurenceMap, predictions);
            TryInference(predictions);
            predictions.Dispose();
            worldOccurence.Dispose();
            occurenceMap.Dispose();
        }
        //Compress byte[] to int32 (4 bytes)
        [BurstCompile]
        private void Compression(
            NativeMultiHashMap<int, int> worldOccurence,
            NativeHashMap<int, int> occurenceMap
         )
        {
            int count = History.Length - NGram + 1;
            for (int i = 0; i < count; i++)
            {
                var sequence = new NativeArray<byte>(4, Allocator.Temp);
                sequence[0] = History[i];
                sequence[1] = History[i + 1];
                sequence[2] = History[i + 2];
                sequence[3] = 0;
                int key = sequence.Reinterpret<int>(UnsafeUtility.SizeOf<byte>())[0];
                sequence[3] = History[i + 3];
                int occurenceKey = sequence.Reinterpret<int>(UnsafeUtility.SizeOf<byte>())[0];
                sequence.Dispose();
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
        }
        [BurstCompile]
        private void TryInference(NativeHashMap<int, int> predictions)
        {
            if (predictions.TryGetValue(Inference, out int result))
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
             NativeMultiHashMap<int, int> worldOccurence,
            NativeHashMap<int, int> occurenceMap,
            NativeHashMap<int, int> predictions
        )
        {
            var keys = worldOccurence.GetKeyArray(Allocator.Temp);
            foreach (var start in keys)
            {
                int prediction = -1;
                int maximum = 0;
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