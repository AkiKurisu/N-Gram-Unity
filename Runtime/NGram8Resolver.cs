using System;
using Unity.Collections;
using Unity.Collections.LowLevel.Unsafe;
using Unity.Jobs;
using UnityEngine;
namespace Kurisu.NGram
{
    /// <summary>
    /// N-Gram resolver for <see cref="NGram8Job"/>
    /// </summary>
    public class NGram8Resolver : INGramResolver, IDisposable
    {
        private NativeArray<byte>? history;
        private NativeArray<byte>? inference;
        public bool Success { get; private set; }
        public int Result { get; private set; }
        private JobHandle jobHandle;
        private NativeArray<double>? result;
        public int NGram { get; set; } = 8;
        public void Dispose()
        {
            history?.Dispose();
            inference?.Dispose();
        }
        public void Resolve(int[] history, int[] inference)
        {
            if (NGram < 2 || NGram > 8)
            {
                Debug.LogError($"{NGram}-Gram is not valid for {nameof(NGram8Resolver)}");
                return;
            }
            Resolve(history, inference, 0, history.Length);
        }
        public void Resolve(int[] history, int[] inference, int historyStartIndex, int historyLength)
        {
            result = new NativeArray<double>(1, Allocator.TempJob);
            var historyArray = new NativeArray<byte>(historyLength, Allocator.TempJob);
            for (int i = 0; i < historyLength; i++)
            {
                historyArray[i] = (byte)history[i + historyStartIndex];
            }
            this.history = historyArray;
            var inferenceArray = new NativeArray<byte>(inference.Length, Allocator.TempJob);
            for (int i = 0; i < inference.Length; i++)
            {
                inferenceArray[i] = (byte)inference[i];
            }
            this.inference = inferenceArray;
            jobHandle = new NGram8Job()
            {
                History = this.history.Value,
                Inference = this.inference.Value,
                Result = result.Value,
                NGram = NGram
            }.Schedule();
        }
        public void Complete()
        {
            jobHandle.Complete();
            Success = result.Value[0] >= 0;
            if (Success)
                Result = result.Value.Reinterpret<byte>(UnsafeUtility.SizeOf<double>())[NGram - 1];
            result.Value.Dispose();
            history.Value.Dispose();
            inference.Value.Dispose();
            inference = null;
            history = null;
            result = null;
        }
    }
}