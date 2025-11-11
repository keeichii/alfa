#!/usr/bin/env python3
"""
Simple test script for preprocessing pipeline.

Tests:
1. Obvious case: normal document processing
2. Edge case: empty fields, missing data
3. Performance: large corpus handling
"""
import json
import sys
import tempfile
from pathlib import Path

# add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.ingest import build_corpus, read_websites_csv
from src.chunker import DocumentChunker
from src.utils import load_jsonl, validate_corpus_format, validate_chunk_format


def test_obvious_case():
    """Test 1: Obvious case - normal document processing."""
    print("=" * 60)
    print("Test 1: Obvious Case - Normal Document Processing")
    print("=" * 60)
    
    # create test CSV
    test_csv = tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False, encoding='utf-8')
    test_csv.write("web_id,url,kind,title,text\n")
    test_csv.write('1,https://test.ru/,html,"Тестовый заголовок","Это тестовый текст страницы с некоторым содержимым."\n')
    test_csv.write('2,https://test2.ru/,html,"Второй заголовок","Еще один текст для проверки обработки."\n')
    test_csv.close()
    
    # create output file
    output_jsonl = tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False, encoding='utf-8')
    output_jsonl.close()
    
    try:
        # build corpus
        build_corpus(test_csv.name, output_jsonl.name)
        
        # verify output
        corpus = load_jsonl(output_jsonl.name)
        assert len(corpus) == 2, f"Expected 2 documents, got {len(corpus)}"
        
        # validate format
        for doc in corpus:
            assert validate_corpus_format(doc), f"Invalid document format: {doc}"
            assert "id" in doc and doc["id"] in ["1", "2"]
            assert "contents" in doc
            assert "\n" in doc["contents"], "Title and text should be separated by newline"
        
        print("✓ Corpus building: PASSED")
        print(f"  - Processed {len(corpus)} documents")
        print(f"  - Sample document: {json.dumps(corpus[0], ensure_ascii=False)[:100]}...")
        
        # test chunking
        chunk_output = tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False, encoding='utf-8')
        chunk_output.close()
        
        chunker = DocumentChunker(chunk_size=10, overlap=2, add_title_prefix=True)
        chunker.chunk_corpus(output_jsonl.name, chunk_output.name)
        
        chunks = load_jsonl(chunk_output.name)
        assert len(chunks) > 0, "Should generate at least one chunk"
        
        for chunk in chunks:
            assert validate_chunk_format(chunk), f"Invalid chunk format: {chunk}"
            assert "doc_id" in chunk
        
        print("✓ Chunking: PASSED")
        print(f"  - Generated {len(chunks)} chunks from 2 documents")
        
    finally:
        Path(test_csv.name).unlink(missing_ok=True)
        Path(output_jsonl.name).unlink(missing_ok=True)
        Path(chunk_output.name).unlink(missing_ok=True)
    
    print()


def test_edge_cases():
    """Test 2: Edge cases - empty fields, missing data."""
    print("=" * 60)
    print("Test 2: Edge Cases - Empty Fields, Missing Data")
    print("=" * 60)
    
    # create test CSV with edge cases
    test_csv = tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False, encoding='utf-8')
    test_csv.write("web_id,url,kind,title,text\n")
    test_csv.write('1,https://test.ru/,html,"",""\n')  # empty title and text
    test_csv.write('2,https://test2.ru/,html,"Только заголовок",""\n')  # empty text
    test_csv.write(',https://test3.ru/,html,"Нет ID","Текст"\n')  # missing web_id
    test_csv.write('4,https://test4.ru/,html,"Нормальный","Нормальный текст"\n')  # normal
    test_csv.close()
    
    output_jsonl = tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False, encoding='utf-8')
    output_jsonl.close()
    
    try:
        build_corpus(test_csv.name, output_jsonl.name)
        corpus = load_jsonl(output_jsonl.name)
        
        # should only have document with id=4 (normal one)
        # others should be skipped
        valid_ids = [doc["id"] for doc in corpus]
        assert "4" in valid_ids, "Normal document should be included"
        assert "1" not in valid_ids or len([d for d in corpus if d["id"] == "1" and not d["contents"].strip()]) == 0, "Empty document should be skipped"
        
        print("✓ Edge case handling: PASSED")
        print(f"  - Processed {len(corpus)} valid documents (skipped invalid ones)")
        
    finally:
        Path(test_csv.name).unlink(missing_ok=True)
        Path(output_jsonl.name).unlink(missing_ok=True)
    
    print()


def test_performance():
    """Test 3: Performance - large corpus handling."""
    print("=" * 60)
    print("Test 3: Performance - Large Corpus Handling")
    print("=" * 60)
    
    import time
    
    # create larger test CSV
    test_csv = tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False, encoding='utf-8')
    test_csv.write("web_id,url,kind,title,text\n")
    
    # generate 100 documents
    for i in range(100):
        title = f"Заголовок {i}"
        text = f"Это тестовый текст документа номер {i}. " * 10  # make it longer
        test_csv.write(f'{i},https://test{i}.ru/,html,"{title}","{text}"\n')
    test_csv.close()
    
    output_jsonl = tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False, encoding='utf-8')
    output_jsonl.close()
    
    try:
        start = time.time()
        build_corpus(test_csv.name, output_jsonl.name)
        build_time = time.time() - start
        
        corpus = load_jsonl(output_jsonl.name)
        assert len(corpus) == 100, f"Expected 100 documents, got {len(corpus)}"
        
        print(f"✓ Performance test: PASSED")
        print(f"  - Processed {len(corpus)} documents in {build_time:.2f} seconds")
        print(f"  - Average: {build_time/len(corpus)*1000:.2f} ms per document")
        
        # test chunking performance
        chunk_output = tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False, encoding='utf-8')
        chunk_output.close()
        
        start = time.time()
        chunker = DocumentChunker(chunk_size=50, overlap=10)
        chunker.chunk_corpus(output_jsonl.name, chunk_output.name)
        chunk_time = time.time() - start
        
        chunks = load_jsonl(chunk_output.name)
        print(f"  - Generated {len(chunks)} chunks in {chunk_time:.2f} seconds")
        print(f"  - Average: {chunk_time/len(corpus)*1000:.2f} ms per document for chunking")
        
    finally:
        Path(test_csv.name).unlink(missing_ok=True)
        Path(output_jsonl.name).unlink(missing_ok=True)
        Path(chunk_output.name).unlink(missing_ok=True)
    
    print()


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("RAG Pipeline Preprocessing Tests")
    print("=" * 60 + "\n")
    
    try:
        test_obvious_case()
        test_edge_cases()
        test_performance()
        
        print("=" * 60)
        print("All tests PASSED!")
        print("=" * 60)
        
    except AssertionError as e:
        print(f"\n❌ Test FAILED: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

