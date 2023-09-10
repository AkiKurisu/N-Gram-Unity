using System.Collections;
using System.Linq;
using UnityEngine;
namespace Kurisu.NGram
{
    public class SkillNGramRunner : MonoBehaviour
    {
        [SerializeField]
        private string[] historySkills;
        [SerializeField]
        private TextAsset tokenizer;
        private INGramResolver resolver;
        private TokenizerReader tokenizerReader;
        private IEnumerator Start()
        {
            resolver = new NGram4Resolver();
            tokenizerReader = new TokenizerReader(tokenizer.text);
            var historyIndex = historySkills.Select(x => tokenizerReader.GetIndex(x)).ToArray();
            resolver.Resolve(historyIndex, historyIndex);
            yield return new WaitForEndOfFrame();
            resolver.Complete();
            Debug.Log("Predict:" + tokenizerReader.GetToken(resolver.Result));
        }
        private void OnDestroy()
        {
            resolver.Dispose();
        }
    }
}
