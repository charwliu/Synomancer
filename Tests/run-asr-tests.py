#!/usr/bin/env python3
"""Test local speech caches, ASR streaming and TTS round trips using Xcode build products."""
import argparse
import os
from pathlib import Path
import subprocess

root = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--derived-data', type=Path, required=True)
parser.add_argument('--configuration', choices=['Debug', 'Release'], default='Debug')
parser.add_argument('--source-packages', type=Path, help='Override the Xcode package checkout directory')
parser.add_argument('--audio', type=Path, help='Non-silent speech WAV/CAF fixture')
parser.add_argument('--cache', type=Path, default=root / '.build/ASRModel')
parser.add_argument('--offline', action='store_true', help='Require a complete cached model; never download')
parser.add_argument('--expected-words', nargs='*', default=[], help='Words that must appear in the transcript')
parser.add_argument('--components-only', action='store_true', help='Test cache resolution and streaming buffers without a model')
parser.add_argument('--chat-muse', action='store_true', help='Test native Muse Glimmer MXFP8 text, images and ATEM output')
parser.add_argument('--muse-components', action='store_true', help='Only run Muse synthetic MXFP8 and ATEM checks without model weights')
parser.add_argument('--chat-moe', action='store_true', help='Test cached Qwen3.5 MoE raw generation and App display')
parser.add_argument('--chat-tokenizer', action='store_true', help='Test missing Gemma PT chat template and local template formats without loading weights')
parser.add_argument('--image-flux2', action='store_true', help='Generate and edit with native FLUX.2 Klein 9B')
parser.add_argument('--image-flux2-dev', action='store_true', help='Generate and edit with cached official FLUX.2-dev BF16 weights')
parser.add_argument('--image-flux1', action='store_true', help='Generate with native FLUX.1-dev and test cancellation/retry')
parser.add_argument('--image-model-benchmark', action='store_true', help='Compare actual FLUX.1 BF16 and Qwen-Image-2.1 BF16/4bit first and repeated requests')
parser.add_argument('--flux1-parity', action='store_true', help='Compare FLUX.1 components and tokenizers with CPU references')
parser.add_argument('--flux2-parity', action='store_true', help='Compare native Klein components with CPU Diffusers fixtures')
parser.add_argument('--flux2-checks', action='store_true', help='Test actual Klein cancellation, restart and unsupported ControlNet rejection')
parser.add_argument('--flux2-dev-parity', action='store_true', help='Compare native FLUX.2-dev Mistral, guidance and tokenizer against CPU references')
parser.add_argument('--image', action='store_true', help='Generate and edit actual Qwen-Image-2.1 images using pure Swift/MLX')
parser.add_argument('--image-parity', action='store_true', help='Compare native image components against CPU Diffusers fixtures')
parser.add_argument('--image-residency', action='store_true', help='Test chat/image reservation cancellation and error recovery without image weights')
parser.add_argument('--image-checks', action='store_true', help='Test cache completeness and native image cancellation/restart')
parser.add_argument('--image-reference-sizes', action='store_true', help='Test actual vision preprocessing and VAE/DiT grid alignment without model weights')
parser.add_argument('--image-negative-prompt', help='Optional true CFG negative prompt for image smoke')
parser.add_argument('--image-steps', type=int, help='Override image smoke denoising steps')
parser.add_argument('--image-size', type=int, choices=[512, 768, 1024], default=512, help='Square size for actual image generation/editing')
parser.add_argument('--image-prefix-benchmark', action='store_true', help='With --image and --image-reference: ABBA comparison of warmed prefix reuse')
parser.add_argument('--image-repeat', action='store_true', help='Repeat editing with identical inputs to verify conditioning reuse')
parser.add_argument('--image-output', type=Path, help='Directory for image artifacts and benchmark comparisons')
parser.add_argument('--image-reference', type=Path, help='Edit this existing photo instead of generating the initial red mug')
parser.add_argument('--image-prompt', help='Override the editing instruction when using --image-reference')
parser.add_argument('--metal-validation', action='store_true', help='Enable Metal API validation, as in Xcode Run')
parser.add_argument('--chat-reasoning', action='store_true', help='Test real Qwen prefilled thinking in ordinary and MTP App streams')
parser.add_argument('--chat-mtp', action='store_true', help='Benchmark real OptiQ MTP and test greedy parity, stop, cancellation and sampling fallback')
parser.add_argument('--chat-library', action='store_true', help='Test upgraded architecture registry, mixed quantization and main-weight selection')
parser.add_argument('--remote-store-only', action='store_true', help='Test remote model discovery and saved conversations without local inference')
parser.add_argument('--remote-api-only', action='store_true', help='Run remote API HTTP fixtures without local inference or personal keys')
parser.add_argument('--chat-api', action='store_true', help='Test local OpenAI-compatible HTTP API and remote client protocol')
parser.add_argument('--chat-store', action='store_true', help='Test production chat stores and completion panel with an in-memory database')
parser.add_argument('--reference-prompt', action='store_true', help='Test image prompt store lifecycle and optionally caption --image-reference with a local vision model')
parser.add_argument('--chat', action='store_true', help='Test local MLX chat history, vision and cancellation')
parser.add_argument('--model', help='Local cached chat model id (default: smallest supported model)')
parser.add_argument('--tts', action='store_true', help='Generate real Chinese/English TTS into .build/TTSTestAudio; no playback')
parser.add_argument('--tts-continuity', action='store_true', help='Compare real codec full/streaming waveforms and sample continuity')
parser.add_argument('--tts-streaming', action='store_true', help='Benchmark batch/streaming TTS and verify streamed audio with ASR')
parser.add_argument('--streaming', action='store_true', help='Use the public 20s Qwen3ASR fixture to verify live partial revisions')
parser.add_argument('--playback', action='store_true', help='Test app synthesis/playback/stop using isolated settings and muted output')
args = parser.parse_args()
if args.chat_muse or args.muse_components:
    args.chat_muse = True
    args.chat = True
if args.remote_store_only:
    args.chat_store = True
if args.remote_api_only:
    args.chat_api = True
if args.image_flux2:
    args.image = True
if args.image_flux2_dev:
    args.image = True
if args.image_flux1:
    args.image = True
if args.image_model_benchmark:
    args.image = True
if args.flux1_parity:
    args.image_checks = True
if args.flux2_parity:
    args.image_checks = True
if args.flux2_checks:
    args.image_checks = True
if args.flux2_dev_parity:
    args.image_checks = True
if args.image_reference_sizes or args.image_residency:
    args.image_checks = True
if args.reference_prompt:
    args.chat = True
    if args.image_reference and not args.model:
        parser.error('--reference-prompt with --image-reference requires --model')
if args.metal_validation:
    os.environ['MTL_DEBUG_LAYER'] = '1'
if not args.components_only and not args.tts and not args.tts_streaming and not args.tts_continuity and not args.playback and not args.chat and not args.chat_store and not args.chat_api and not args.chat_library and not args.chat_mtp and not args.chat_reasoning and not args.chat_moe and not args.chat_tokenizer and not args.image and not args.image_parity and not args.image_checks and args.audio is None:
    parser.error('--audio is required for real inference')
products = args.derived_data.resolve() / 'Build/Products' / args.configuration
checkouts = (args.source_packages or (args.derived_data / 'SourcePackages')).resolve() / 'checkouts'
maps = args.derived_data.resolve() / 'Build/Intermediates.noindex/GeneratedModuleMaps'
objects = '''Qwen3TTS Qwen3ASR AudioCommon MLXCommon MLXFFT MLX MLXNN MLXFast
Cmlx cLogSupport Numerics ComplexModule RealModule _NumericsShims Hub Tokenizers
Jinja OrderedCollections DequeModule InternalCollectionsUtilities yyjson
HuggingFace Crypto Logging EventSource'''.split()
executable = products / 'SynomancerASRSmoke'
sdk = subprocess.check_output(['xcrun', '--show-sdk-path'], text=True).strip()
command = ['swiftc', '-swift-version', '5', '-parse-as-library',
           '-O' if args.configuration == 'Release' else '-Onone',
           '-module-cache-path', str(root / '.build/TestModuleCache'),
           '-target', 'arm64-apple-macos27.0', '-sdk', sdk, '-I', str(products),
           '-I', str(checkouts / 'mlx-swift/Source/Cmlx/include'),
           '-I', str(checkouts / 'mlx-swift/Source/Cmlx/mlx-c'),
           '-I', str(checkouts / 'swift-numerics/Sources/_NumericsShims/include')]
for module in ('cLogSupport', 'yyjson'):
    command += ['-Xcc', '-fmodule-map-file=' + str(maps / (module + '.modulemap'))]
if args.image or args.image_parity or args.image_checks:
    command[command.index('-swift-version') + 1] = '6'
    objects += ['MLXLLM', 'MLXVLM', 'MLXLMCommon', 'MLXOptimizers']
    command += [str(root / 'Tests/LocalImageSmoke.swift'), str(root / 'Tests/ImageModelPerformanceTests.swift'), str(root / 'Tests/Flux1Tests.swift'), str(root / 'Tests/Flux2KleinTests.swift'), str(root / 'Tests/Flux2DevTests.swift'), str(root / 'Applications/Synomancer/Services/SpeechCacheAccess.swift')]
    command += [str(path) for path in sorted((root / 'Applications/Synomancer/Services/ImageGeneration').glob('*.swift'))]
    # Image generation shares the chat service's model-residency gate.
    command += [str(root / path) for path in [
        'Applications/Synomancer/Services/LocalChatService.swift', 'Applications/Synomancer/Services/LocalMuseGlimmerOutput.swift',
        'Applications/Synomancer/Services/LocalChatModelCache.swift',
        'Applications/Synomancer/Services/LocalChatTokenizerLoader.swift',
        'Applications/Synomancer/Services/LocalQwenMoEModel.swift',
        'Applications/Synomancer/Services/LocalQwenMTP.swift', 'Applications/Synomancer/Services/LocalQwenDenseOptimization.swift',
        'Applications/Synomancer/Services/AssistantMessageContent.swift']]
    command += [str(path) for path in sorted((root / 'Applications/Synomancer/Services/GGUF').glob('*.swift'))]
elif args.chat or args.chat_store or args.chat_api or args.chat_library or args.chat_mtp or args.chat_reasoning or args.chat_moe or args.chat_tokenizer:
    objects += ['MLXLLM', 'MLXVLM', 'MLXLMCommon', 'MLXOptimizers']
    command += [str(root / path) for path in [
        'Tests/LocalMuseGlimmerTests.swift' if args.chat_muse else 'Tests/ReferenceImagePromptTests.swift' if args.reference_prompt else 'Tests/LocalChatTokenizerTests.swift' if args.chat_tokenizer else 'Tests/LocalChatMoESmoke.swift' if args.chat_moe else 'Tests/LocalChatReasoningSmoke.swift' if args.chat_reasoning else 'Tests/LocalChatMTPSmoke.swift' if args.chat_mtp else 'Tests/MLXModelLibraryTests.swift' if args.chat_library else ('Tests/ChatAPITests.swift' if args.chat_api else ('Tests/LocalChatStoreTests.swift' if args.chat_store else 'Tests/LocalChatSmoke.swift')), 'Applications/Synomancer/Services/LocalChatModelCache.swift',
        'Applications/Synomancer/Services/AssistantMessageContent.swift', 'Applications/Synomancer/Services/LocalQwenMoEModel.swift', 'Applications/Synomancer/Services/LocalChatService.swift', 'Applications/Synomancer/Services/LocalMuseGlimmerOutput.swift', 'Applications/Synomancer/Services/LocalQwenMTP.swift', 'Applications/Synomancer/Services/LocalQwenDenseOptimization.swift', 'Applications/Synomancer/Services/LocalChatTokenizerLoader.swift', 'Applications/Synomancer/Services/SpeechCacheAccess.swift']]
    command += [str(path) for path in sorted((root / 'Applications/Synomancer/Services/GGUF').glob('*.swift'))]
    if args.reference_prompt:
        command[command.index('-swift-version') + 1] = '6'
        command += [str(root / path) for path in [
            'Applications/Synomancer/Services/LocalReferencePromptService.swift',
            'Applications/Synomancer/Stores/ImageStudioStore.swift']]
        command += [str(path) for path in sorted((root / 'Applications/Synomancer/Services/ImageGeneration').glob('*.swift'))]
    if args.chat_store or args.chat_api:
        command += [str(root / path) for path in [
            'Applications/Synomancer/Services/OpenAIConfiguration.swift',
            'Applications/Synomancer/Services/OpenAIChatService.swift']]
    if args.chat_api:
        command += [str(root / 'Applications/Synomancer/Services/LocalChatAPIServer.swift')]
    if args.chat_store:
        command += [str(root / 'Applications/Synomancer/Services/ChatGenerationService.swift'), str(root / 'Applications/Synomancer/Services/AutomaticResponseSpeech.swift')]
        objects += ['Synogging']
        command += [str(args.derived_data.resolve() / 'Build/Intermediates.noindex/Synomancer.build' / args.configuration / 'Synomancer.build/DerivedSources/GeneratedAssetSymbols.swift')]
        command += [str(path) for path in sorted((root / 'Applications/Synomancer/Services/DataService').glob('*.swift'))]
        command += [str(root / path) for path in [
            'Applications/Synomancer/Stores/ConversationStore.swift', 'Applications/Synomancer/Stores/LanguageModelStore.swift',
            'Applications/Synomancer/Stores/AppStore.swift', 'Applications/Synomancer/Models/LanguageModel.swift',
            'Applications/Synomancer/Models/ConversationState.swift', 'Applications/Synomancer/Models/NotificationMessage.swift',
            'Applications/Synomancer/SwiftData/Models/ConversationSD.swift', 'Applications/Synomancer/SwiftData/Models/MessageSD.swift',
            'Applications/Synomancer/SwiftData/Models/LanguageModelSD.swift', 'Applications/Synomancer/SwiftData/Models/CompletionInstructionSD.swift',
            'Applications/Synomancer/Extensions/ModelContext+Extension.swift', 'Applications/Synomancer/Utilities/Logging.swift',
            'Applications/Synomancer/Extensions/View+Extension.swift', 'Applications/Synomancer/Extensions/UIImage+Extension.swift',
            'Applications/Synomancer/Extensions/Colours+Extension.swift', 'Applications/Synomancer/UI/Helpers.swift',
            'Applications/Synomancer/UI/macOS/PromptPanel/PromptPanelVM.swift', 'Applications/Synomancer/Helpers/AsyncQueue.swift',
        ]]
elif args.playback:
    command += [str(root / path) for path in [
        'Tests/SpeechPlaybackTests.swift', 'Applications/Synomancer/Services/AutomaticResponseSpeech.swift',
        'Applications/Synomancer/Services/AssistantMessageContent.swift', 'Applications/Synomancer/Services/SpeechPlaybackTail.swift',
        'Applications/Synomancer/Services/SpeechService.swift',
        'Applications/Synomancer/Services/LocalTTSService.swift', 'Applications/Synomancer/Services/SpeechCacheAccess.swift']]
else:
    command += [str(root / ('Tests/TTSCodecContinuityTests.swift' if args.tts_continuity else 'Tests/LocalTTSStreamingSmoke.swift' if args.tts_streaming else ('Tests/LocalTTSSmoke.swift' if args.tts else ('Tests/ASRCacheAndStreamingTests.swift' if args.components_only else 'Tests/LocalASRSmoke.swift'))))]
command += [str(products / (name + '.o')) for name in objects]
command += ['-lc++', '-framework', 'Vision', '-framework', 'CoreImage', '-framework', 'ImageIO', '-framework', 'Accelerate', '-framework', 'Metal',
            '-framework', 'Foundation', '-framework', 'CoreML',
            '-framework', 'AVFoundation', '-o', str(executable)]
subprocess.run(command, check=True, cwd=root)
if args.flux2_dev_parity:
    subprocess.run([str(executable), '--flux2-dev-parity', str(root / '.build/Flux2DevParity'), str(root / '.build/Flux2DevMetadata')], check=True, cwd=products)
elif args.image_residency:
    subprocess.run([str(executable), '--residency-gate'], check=True, cwd=products)
elif args.image_reference_sizes:
    subprocess.run([str(executable), '--reference-sizes'], check=True, cwd=products)
elif args.flux1_parity:
    subprocess.run([str(executable), str(args.cache.resolve()), str(root / '.build/Flux1Parity'), '--flux1-parity'], check=True, cwd=products)
elif args.flux2_parity:
    subprocess.run([str(executable), '--flux2-parity', str(root / '.build/Flux2Parity')], check=True, cwd=products)
elif args.flux2_checks:
    subprocess.run([str(executable), str(args.cache.resolve()), str(root / '.build/Flux2Images'), '--flux2-checks'], check=True, cwd=products)
elif args.image_checks:
    subprocess.run([str(executable), str(args.cache.resolve()), str(root / '.build/ImageTestOutput'), '--checks'], check=True, cwd=products)
elif args.image_parity:
    subprocess.run([str(executable), str(args.cache.resolve()), str(root / '.build/ImageParity'), '--parity'], check=True, cwd=products)
elif args.image:
    image_command = [str(executable), str(args.cache.resolve()), str((args.image_output or root / '.build/ImageTestOutput').resolve()), '--size', str(args.image_size)]
    if args.image_negative_prompt:
        image_command += ["--negative-prompt", args.image_negative_prompt]
    if args.image_steps:
        image_command += ["--steps", str(args.image_steps)]
    if args.image_flux2:
        image_command += ['--flux2']
    if args.image_flux2_dev:
        image_command += ['--flux2-dev']
    if args.image_flux1:
        image_command += ['--flux1']
    if args.image_model_benchmark:
        image_command += ['--model-benchmark']
    if args.image_prefix_benchmark:
        image_command += ['--prefix-benchmark']
    if args.image_repeat:
        image_command += ['--repeat']
    if args.image_reference:
        image_command += ['--reference', str(args.image_reference.resolve())]
    if args.image_prompt:
        image_command += ['--prompt', args.image_prompt]
    subprocess.run(image_command, check=True, cwd=products)
elif args.chat or args.chat_store or args.chat_api or args.chat_library or args.chat_mtp or args.chat_reasoning or args.chat_moe or args.chat_tokenizer:
    subprocess.run([str(executable), str(args.cache.resolve()), *([args.model] if args.model else []), *(["--components-only"] if args.muse_components else []), *(["--remote-only"] if args.remote_api_only or args.remote_store_only else []), *([str(args.image_reference.resolve())] if args.reference_prompt and args.image_reference else [])], check=True, cwd=products)
elif args.playback:
    subprocess.run([str(executable), str(args.cache.resolve())], check=True, cwd=products)
elif args.tts or args.tts_streaming or args.tts_continuity:
    subprocess.run([str(executable), str(args.cache.resolve()), str(root / ".build/TTSTestAudio")], check=True, cwd=products)
elif args.components_only:
    subprocess.run([str(executable)], check=True, cwd=products)
else:
    subprocess.run([str(executable), str(args.cache.resolve()), str(args.audio.resolve()),
                    *(["--offline"] if args.offline else []), *(["--streaming"] if args.streaming else []), *args.expected_words],
                   check=True, cwd=products)
