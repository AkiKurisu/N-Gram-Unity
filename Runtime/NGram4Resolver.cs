using System;
using Unity.Collections;
using Unity.Jobs;
namespace Kurisu.NGram
{
    /// <summary>
    /// An example N-Gram resolver for 4-Gram
    /// </summary>
    public class NGram4Resolver : IDisposable
    {
        private NativeArray<byte>? history;
        private const int NGram = 4;
        public byte Result { get; private set; }
        private JobHandle jobHandle;
        private NativeArray<int>? result;
        public void Dispose()
        {
            history?.Dispose();
        }
        public void Resolve(byte[] history, byte[] inference)
        {
            result?.Dispose();
            result = new NativeArray<int>(1, Allocator.TempJob);
            this.history?.Dispose();
            this.history = new NativeArray<byte>(history, Allocator.TempJob);
            byte word1 = inference[^3];
            byte word2 = inference[^2];
            byte word3 = inference[^1];
            //Zip in a simple int value
            int inferenceKey = BitConverter.ToInt32(new byte[] { word1, word2, word3, 0 });
            jobHandle = new NGramJob()
            {
                History = this.history.Value,
                Inference = inferenceKey,
                Result = result.Value,
                NGram = NGram
            }.Schedule();
        }
        public void Complete()
        {
            jobHandle.Complete();
            //[first][second][third][predict]
            Result = BitConverter.GetBytes(result.Value[0])[3];
            result.Value.Dispose();
            history.Value.Dispose();
            history = null;
            result = null;
        }
    }
}