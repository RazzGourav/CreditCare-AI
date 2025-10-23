async function postPredict(payload) {
        const resp = await fetch('/predict', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(payload)
        });
        return await resp.json();
      }

      function readFormValues() {
        const form = document.getElementById('predict-form');
        const fd = new FormData(form);
        const out = {};
        for (const [k, v] of fd.entries()) {
          out[k] = v === '' ? null : Number(v);
        }
        return out;
      }

      function setLoadingState(loading = true) {
        const btn = document.getElementById('predict-btn');
        if (!btn) return;
        if (loading) {
          btn.disabled = true;
          btn.innerHTML = '<span>⏳ Analyzing...</span>';
          btn.style.opacity = '0.7';
        } else {
          btn.disabled = false;
          btn.innerHTML = '<span>🎯 Analyze Risk Profile</span>';
          btn.style.opacity = '1';
        }
      }

      function showResult(json) {
        const probText = document.getElementById('prob-text');
        const meterBar = document.getElementById('meter-bar');
        const classLabel = document.getElementById('class-label');
        const explain = document.getElementById('explain');

        if (json.error) {
          probText.innerText = 'Error';
          explain.innerText = json.error;
          meterBar.style.width = '0%';
          classLabel.innerHTML = '<span class="risk-label">⚠️ Error occurred</span>';
          return;
        }

        const p = Math.round((json.probability || 0) * 1000) / 10;
        probText.innerText = (isNaN(p) ? '—' : `${p}%`);
        
        const width = Math.min(100, Math.max(2, (json.probability || 0) * 100));
        meterBar.style.width = width + '%';
        
        if (json.prediction === 1) {
          classLabel.innerHTML = '<span class="risk-label high">⚠️ High Risk - Likely to Default</span>';
        } else {
          classLabel.innerHTML = '<span class="risk-label low">✅ Low Risk - Unlikely to Default</span>';
        }
        
        explain.innerText = json.explanation || 'Risk assessment complete. Review the probability score above.';
      }

      // Event Listeners
      document.addEventListener('DOMContentLoaded', () => {
        const predictBtn = document.getElementById('predict-btn');
        const exampleBtn = document.getElementById('example-btn');
        const resetBtn = document.getElementById('reset-btn');

        if (predictBtn) {
          predictBtn.addEventListener('click', async () => {
            const payload = readFormValues();
            setLoadingState(true);
            try {
              const json = await postPredict(payload);
              showResult(json);
            } catch (err) {
              showResult({ error: err.message || String(err) });
            } finally {
              setLoadingState(false);
            }
          });
        }

        if (exampleBtn) {
          exampleBtn.addEventListener('click', () => {
            const defaults = {
              LIMIT_BAL: 20000, AGE: 30, SEX: 1, EDUCATION: 2, MARRIAGE: 1,
              PAY_0: 0, PAY_2: 0, PAY_3: 0, PAY_4: 0, PAY_5: 0, PAY_6: 0,
              BILL_AMT1: 3913, BILL_AMT2: 3102, BILL_AMT3: 689, 
              BILL_AMT4: 0, BILL_AMT5: 0, BILL_AMT6: 0,
              PAY_AMT1: 0, PAY_AMT2: 689, PAY_AMT3: 0, 
              PAY_AMT4: 0, PAY_AMT5: 0, PAY_AMT6: 0
            };
            const form = document.getElementById('predict-form');
            for (const k in defaults) {
              if (form.elements[k]) form.elements[k].value = defaults[k];
            }
          });
        }

        if (resetBtn) {
          resetBtn.addEventListener('click', () => {
            document.getElementById('predict-form').reset();
            document.getElementById('prob-text').innerText = '—';
            document.getElementById('meter-bar').style.width = '0%';
            document.getElementById('class-label').innerHTML = '—';
            document.getElementById('explain').innerText = 'Enter client data and click "Analyze Risk Profile" to generate a prediction.';
          });
        }

        // Add smooth scroll behavior
        document.querySelectorAll('a[href^="#"]').forEach(anchor => {
          anchor.addEventListener('click', function (e) {
            e.preventDefault();
            const target = document.querySelector(this.getAttribute('href'));
            if (target) {
              target.scrollIntoView({ behavior: 'smooth', block: 'start' });
            }
          });
        });
      });