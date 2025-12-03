import React, { useState, useEffect } from 'react';

function App() {
  const [codeQuery, setCodeQuery] = useState('');
  const [synonymQuery, setSynonymQuery] = useState('');
  const [descQuery, setDescQuery] = useState('');
  const [results, setResults] = useState([]);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    // Debounce and only search if any input is given
    if (!codeQuery && !synonymQuery && !descQuery) {
      setResults([]);
      return;
    }

    setLoading(true);

    // Choose which query to send based on priority or whichever filled
    const params = new URLSearchParams();
    if (codeQuery) params.append('code', codeQuery);
    else if (synonymQuery) params.append('synonym', synonymQuery);
    else if (descQuery) params.append('description', descQuery);

    fetch(`http://localhost:8000/search?${params.toString()}`)
      .then(res => res.json())
      .then(data => setResults(data || []))
      .catch(err => {
        console.error('API Error:', err);
        setResults([]);
      })
      .finally(() => setLoading(false));
  }, [codeQuery, synonymQuery, descQuery]);

  return (
    <div style={{ margin: '20px', fontFamily: 'Arial' }}>
      <h2>Ayush Disease Search (NAMASTE + ICD-11 TM2)</h2>

      <input
        placeholder="Search by Code"
        value={codeQuery}
        onChange={e => {
          setCodeQuery(e.target.value);
          setSynonymQuery('');
          setDescQuery('');
        }}
        style={{ marginBottom: 8, width: 400, padding: 6 }}
      />
      <input
        placeholder="Search by Synonym"
        value={synonymQuery}
        onChange={e => {
          setSynonymQuery(e.target.value);
          setCodeQuery('');
          setDescQuery('');
        }}
        style={{ marginBottom: 8, width: 400, padding: 6 }}
      />
      <input
        placeholder="Search by Description"
        value={descQuery}
        onChange={e => {
          setDescQuery(e.target.value);
          setCodeQuery('');
          setSynonymQuery('');
        }}
        style={{ marginBottom: 8, width: 400, padding: 6 }}
      />

      <div style={{ marginTop: 8 }}>
        {loading ? <small>Searching...</small> : <small>{results.length} results</small>}
      </div>

      <ul>
        {results.map((item, idx) => (
          <li key={idx}>
            <b>{item.NAMASTE_code || '-'}</b>: {item.NAMASTE_text}
            <br />
            ICD: <b>{item.closest_ICD11_code || '-'}</b> - {item.closest_ICD11_name || '-'}
            <br />
            Similarity: {item.similarity_score ? item.similarity_score.toFixed(3) : '-'}
          </li>
        ))}
      </ul>
    </div>
  );
}
export default App;
