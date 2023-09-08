using System.Collections;
using System.Collections.Generic;
using UnityEngine;
namespace Kurisu.NGram
{
    //4-Gram implement for text prediction
    public class TextNGramRunner : MonoBehaviour
    {
        [SerializeField, Multiline]
        private string history;
        [SerializeField, Multiline]
        private string inference;
        private byte[] historyIndex;
        private byte[] inferenceIndex;
        private readonly Dictionary<char, byte> char2Byte = new();
        private readonly Dictionary<byte, char> byte2Char = new();
        private NGram4Resolver resolver;
        private byte index = 0;
        private IEnumerator Start()
        {
            Tokenization();
            resolver = new NGram4Resolver();
            resolver.Resolve(historyIndex, inferenceIndex);
            yield return new WaitForEndOfFrame();
            resolver.Complete();
            Debug.Log("Predict:" + byte2Char[resolver.Result]);
        }
        private void Tokenization()
        {
            historyIndex = new byte[history.Length];
            inferenceIndex = new byte[inference.Length];
            for (int i = 0; i < history.Length; i++)
            {
                if (!char2Byte.ContainsKey(history[i]))
                {
                    char2Byte[history[i]] = index;
                    byte2Char[index] = history[i];
                    index++;
                }
                historyIndex[i] = char2Byte[history[i]];
            }
            for (int i = 0; i < inference.Length; i++)
            {
                if (!char2Byte.ContainsKey(inference[i]))
                {
                    char2Byte[inference[i]] = index;
                    byte2Char[index] = inference[i];
                    index++;
                }
                inferenceIndex[i] = char2Byte[inference[i]];
            }
        }
        private void OnDestroy()
        {
            resolver.Dispose();
        }

    }
}