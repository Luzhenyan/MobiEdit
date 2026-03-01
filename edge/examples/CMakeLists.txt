add_library(MLLM_CORE
    ${DIR_SRC_MEM_MANAGER} 
    ${DIR_SRC_EXP} 
    ${DIR_SRC}
    ${DIR_SRC_LEGACY}
    ${PROJECT_SOURCE_DIR}/src/tokenizers/Tokenizer.cpp
    ${PROJECT_SOURCE_DIR}/src/tokenizers/BPE/Bpe.cpp
    ${PROJECT_SOURCE_DIR}/src/tokenizers/WordPiece/WordPiece.cpp
    ${PROJECT_SOURCE_DIR}/src/tokenizers/Tiktoken/tiktoken.cpp
    ${PROJECT_SOURCE_DIR}/src/tokenizers/Unicode.cpp
    ${PROJECT_SOURCE_DIR}/src/tokenizers/UnicodeData.cpp
    ${PROJECT_SOURCE_DIR}/src/processor/PreProcess.cpp
)

macro(func_link_libaries target)
    target_link_libraries(${target} PUBLIC MLLM_CORE MLLM_CPU fmt::fmt-header-only)
    target_link_libraries(${target} PRIVATE nlohmann_json::nlohmann_json)
    if (MLLM_OPENMP)
        target_compile_options(${target} PRIVATE -fopenmp)
        if (ARM)
            target_link_libraries(${target} PUBLIC -fopenmp -static-openmp)
        else ()
            target_link_libraries(${target} PUBLIC -fopenmp)
        endif ()
    endif ()
    if (QNN)
        target_compile_definitions(${target} PUBLIC USE_QNN)
        target_link_libraries(${target} PUBLIC MLLM_QNN)
    endif()
    if (MLLM_BUILD_XNNPACK_BACKEND)
        target_link_libraries(${target} PRIVATE MllmXnnpackBackend)
    endif()
endmacro()

macro(func_llm_add_executable target)
    add_executable(${target}
        ${PROJECT_SOURCE_DIR}/examples/${target}.cpp 

        ${PROJECT_SOURCE_DIR}/src/tokenizers/Tiktoken/tiktoken.cpp
    )
    func_link_libaries(${target})
endmacro()

macro(func_vlm_add_executable target)
    add_executable(${target} 
        ${PROJECT_SOURCE_DIR}/examples/${target}.cpp 

        ${PROJECT_SOURCE_DIR}/src/tokenizers/Unigram/Unigram.cpp
        ${DIR_SRC_PROCESSOE}
        ${DIR_THIRDPARTY_AUDIO}
    )
    func_link_libaries(${target})
endmacro()

# func_llm_add_executable(mllm_benchmark)
# func_llm_add_executable(demo_llama)
# func_llm_add_executable(demo_tinyllama)
# func_llm_add_executable(demo_stablelm)
# func_llm_add_executable(demo_sparse_llama)
# func_llm_add_executable(demo_elastic_llama)
# func_llm_add_executable(demo_gemma)
# func_llm_add_executable(demo_gemma2)
# func_llm_add_executable(demo_qwen)
# func_llm_add_executable(demo_qwen3)
# func_llm_add_executable(demo_ds_qwen2)
# func_llm_add_executable(demo_mistral)
# func_llm_add_executable(demo_yi)
# func_llm_add_executable(demo_opt)
# func_llm_add_executable(demo_phi3)
# func_llm_add_executable(demo_minicpm)
# func_llm_add_executable(demo_minicpm3)
# func_llm_add_executable(demo_minicpm_moe)
# func_llm_add_executable(demo_smollm)
# func_llm_add_executable(demo_openelm)
# func_llm_add_executable(demo_dclm)
# func_llm_add_executable(demo_bert)
# func_llm_add_executable(demo_phonelm)
# func_llm_add_executable(demo_llama3)
# func_llm_add_executable(demo_minicpm_moe_mbm)
# func_llm_add_executable(demo_qwen_sd)
# func_llm_add_executable(demo_minicpm_moe_mbp)


# func_vlm_add_executable(demo_llava)
# func_vlm_add_executable(demo_fuyu)
# func_vlm_add_executable(demo_vit)
# func_vlm_add_executable(demo_clip)
# func_vlm_add_executable(demo_imagebind)
# func_vlm_add_executable(demo_imagebind_1mod)
# func_vlm_add_executable(demo_phi3v)
# func_vlm_add_executable(demo_qwen2_vl)
# func_vlm_add_executable(demo_showui)
# # func_vlm_add_executable(demo)
    func_llm_add_executable(demo_qwen_rome_fwd)

# QNN demo

if(QNN)
    func_llm_add_executable(demo_qwen_npu)
    # func_llm_add_executable(main_qwen_npu)
    # func_llm_add_executable(demo_phonelm_npu)
    # func_llm_add_executable(main_phonelm_npu)
    # func_llm_add_executable(demo_qwen_pipeline)
    # func_llm_add_executable(demo_qwen_rome_fwd)

    # single operator TEST examples
    # func_llm_add_executable(test_qnn)
    # func_llm_add_executable(test_split)
    # func_llm_add_executable(test_rope)
    # func_llm_add_executable(test_softmax)
    # func_llm_add_executable(test_quickgelu)

    # func_vlm_add_executable(demo_qwen2_vl_npu)
    # func_vlm_add_executable(demo_qwen2_vl_npuvit)
endif()


if(MLLM_BUILD_XNNPACK_BACKEND)
    func_llm_add_executable(demo_qwen_xp)
endif()

