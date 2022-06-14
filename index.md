---
layout: default
title: Estonian TTS samples
---

# Estonian Multispeaker TTS with TransformerTTS

These samples are created using single-speaker and multispeaker (GST) models on 10 Estonian speakers. The waveforms are
generated with HiFiGAN.

<table>
<thead>
  <tr>
    <th>Ground truth</th>
    <th>Ground truth (mel + vocoder)</th>
    <th>Baseline (student-teacher)</th>
    <th>Ext. alignments (single-speaker)</th>
    <th>Ext. alignments (multi-speaker, GST)</th>
    <th>Ext. alignments (multi-speaker, embedding)</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td colspan="5">
      <br>
      Albert: <i>Teine põhjus meelemürke inimese nägemisväljast eemal hoida seostub meie lastega.</i>
    </td>
  </tr>
  <tr>
    <td><audio src="audio/gt-albert.wav?raw=true"  controls preload></audio></td>
    <td><audio src="audio/gt-voc-albert.wav?raw=true"  controls preload></audio></td>
    <td><audio src="audio/grapheme-albert.wav?raw=true"  controls preload></audio></td>
    <td><audio src="audio/grapheme-kaldi-albert.wav?raw=true"  controls preload></audio></td>
    <td><audio src="audio/grapheme-kaldi-gst-albert.wav?raw=true"  controls preload></audio></td>
    <td><audio src="audio/grapheme-kaldi-emb-albert.wav?raw=true"  controls preload></audio></td>
  </tr>
  <tr>
    <td colspan="5">
      <br>
      Indrek: <i>Ma õppisin teoloogiat, aga mitte selleks, et kantslisse tõusta ja jutlusi pidada.</i>
    </td>
  </tr>
  <tr>
    <td><audio src="audio/gt-indrek.wav?raw=true"  controls preload></audio></td>
    <td><audio src="audio/gt-voc-indrek.wav?raw=true"  controls preload></audio></td>
    <td><audio src="audio/grapheme-indrek.wav?raw=true"  controls preload></audio></td>
    <td><audio src="audio/grapheme-kaldi-indrek.wav?raw=true"  controls preload></audio></td>
    <td><audio src="audio/grapheme-kaldi-gst-indrek.wav?raw=true"  controls preload></audio></td>
    <td><audio src="audio/grapheme-kaldi-emb-indrek.wav?raw=true"  controls preload></audio></td>
  </tr>
  <tr>
    <td colspan="5">
      <br>
      Kalev: <i>Ma tahan saada terveks ja loodan peagi tööle naasta", rääkis ta toona.</i>
    </td>
  </tr>
  <tr>
    <td><audio src="audio/gt-kalev.wav?raw=true"  controls preload></audio></td>
    <td><audio src="audio/gt-voc-kalev.wav?raw=true"  controls preload></audio></td>
    <td><audio src="audio/grapheme-kalev.wav?raw=true"  controls preload></audio></td>
    <td><audio src="audio/grapheme-kaldi-kalev.wav?raw=true"  controls preload></audio></td>
    <td><audio src="audio/grapheme-kaldi-gst-kalev.wav?raw=true"  controls preload></audio></td>
    <td><audio src="audio/grapheme-kaldi-emb-kalev.wav?raw=true"  controls preload></audio></td>
  </tr>
  <tr>
    <td colspan="5">
      <br>
      Külli: <i>See lihtsalt pidi niiviisi olema, sest kõik muu mu ümber tundus nii tõeline.</i>
    </td>
  </tr>
  <tr>
    <td><audio src="audio/gt-kylli.wav?raw=true"  controls preload></audio></td>
    <td><audio src="audio/gt-voc-kylli.wav?raw=true"  controls preload></audio></td>
    <td><audio src="audio/grapheme-kylli.wav?raw=true"  controls preload></audio></td>
    <td><audio src="audio/grapheme-kaldi-kylli.wav?raw=true"  controls preload></audio></td>
    <td><audio src="audio/grapheme-kaldi-gst-kylli.wav?raw=true"  controls preload></audio></td>
    <td><audio src="audio/grapheme-kaldi-emb-kylli.wav?raw=true"  controls preload></audio></td>
  </tr>
  <tr>
    <td colspan="5">
      <br>
      Liivika: <i>Lamasin öösel unetult Šarlote voodis, teki olin kõrvale heitnud, kuna oli liiga soe.</i>
    </td>
  </tr>
  <tr>
    <td><audio src="audio/gt-liivika.wav?raw=true"  controls preload></audio></td>
    <td><audio src="audio/gt-voc-liivika.wav?raw=true"  controls preload></audio></td>
    <td><audio src="audio/grapheme-liivika.wav?raw=true"  controls preload></audio></td>
    <td><audio src="audio/grapheme-kaldi-liivika.wav?raw=true"  controls preload></audio></td>
    <td><audio src="audio/grapheme-kaldi-gst-liivika.wav?raw=true"  controls preload></audio></td>
    <td><audio src="audio/grapheme-kaldi-emb-liivika.wav?raw=true"  controls preload></audio></td>
  </tr>
  <tr>
    <td colspan="5">
      <br>
      Mari: <i>"Seda lihtsam on meil võimalik lahendada see kitsaskoht, vähendades nii mõnegi ajateenija olmemuresid ning aidates neil rohkem keskenduda väljaõppele", ütles Rannaveski.</i>
    </td>
  </tr>
  <tr>
    <td><audio src="audio/gt-mari.wav?raw=true"  controls preload></audio></td>
    <td><audio src="audio/gt-voc-mari.wav?raw=true"  controls preload></audio></td>
    <td><audio src="audio/grapheme-mari.wav?raw=true"  controls preload></audio></td>
    <td><audio src="audio/grapheme-kaldi-mari.wav?raw=true"  controls preload></audio></td>
    <td><audio src="audio/grapheme-kaldi-gst-mari.wav?raw=true"  controls preload></audio></td>
    <td><audio src="audio/grapheme-kaldi-emb-mari.wav?raw=true"  controls preload></audio></td>
  </tr>
  <tr>
    <td colspan="5">
      <br>
      Meelis: <i>"Oru Pearul on ometi õigus, kui ta ütleb, et kool kasvatab hobusevargaid."</i>
    </td>
  </tr>
  <tr>
    <td><audio src="audio/gt-meelis.wav?raw=true"  controls preload></audio></td>
    <td><audio src="audio/gt-voc-meelis.wav?raw=true"  controls preload></audio></td>
    <td><audio src="audio/grapheme-meelis.wav?raw=true"  controls preload></audio></td>
    <td><audio src="audio/grapheme-kaldi-meelis.wav?raw=true"  controls preload></audio></td>
    <td><audio src="audio/grapheme-kaldi-gst-meelis.wav?raw=true"  controls preload></audio></td>
    <td><audio src="audio/grapheme-kaldi-emb-meelis.wav?raw=true"  controls preload></audio></td>
  </tr>
  <tr>
    <td colspan="5">
      <br>
      Peeter: <i>Kulunud pruuni ülikonda kandev mees toetas küünarnukid lauale, sättis käelaba lõuale toeks ja hakkas teda jõllitama.</i>
    </td>
  </tr>
  <tr>
    <td><audio src="audio/gt-peeter.wav?raw=true"  controls preload></audio></td>
    <td><audio src="audio/gt-voc-peeter.wav?raw=true"  controls preload></audio></td>
    <td><audio src="audio/grapheme-peeter.wav?raw=true"  controls preload></audio></td>
    <td><audio src="audio/grapheme-kaldi-peeter.wav?raw=true"  controls preload></audio></td>
    <td><audio src="audio/grapheme-kaldi-gst-peeter.wav?raw=true"  controls preload></audio></td>
    <td><audio src="audio/grapheme-kaldi-emb-peeter.wav?raw=true"  controls preload></audio></td>
  </tr>
  <tr>
    <td colspan="5">
      <br>
      Tambet: <i>Nii sidus eesliajaja looma esijalad kokku, võttis koorma endale selga ja läks tagasi oma üüritud kohta värava juures.</i>
    </td>
  </tr>
  <tr>
    <td><audio src="audio/gt-tambet.wav?raw=true"  controls preload></audio></td>
    <td><audio src="audio/gt-voc-tambet.wav?raw=true"  controls preload></audio></td>
    <td><audio src="audio/grapheme-tambet.wav?raw=true"  controls preload></audio></td>
    <td><audio src="audio/grapheme-kaldi-tambet.wav?raw=true"  controls preload></audio></td>
    <td><audio src="audio/grapheme-kaldi-gst-tambet.wav?raw=true"  controls preload></audio></td>
    <td><audio src="audio/grapheme-kaldi-emb-tambet.wav?raw=true"  controls preload></audio></td>
  </tr>
  <tr>
    <td colspan="5">
      <br>
      Vesta: <i>Juba homme läheb Tallinna Lauluväljakul suuremaks võidukihutamiseks!</i>
    </td>
  </tr>
  <tr>
    <td><audio src="audio/gt-vesta.wav?raw=true"  controls preload></audio></td>
    <td><audio src="audio/gt-voc-vesta.wav?raw=true"  controls preload></audio></td>
    <td><audio src="audio/grapheme-vesta.wav?raw=true"  controls preload></audio></td>
    <td><audio src="audio/grapheme-kaldi-vesta.wav?raw=true"  controls preload></audio></td>
    <td><audio src="audio/grapheme-kaldi-gst-vesta.wav?raw=true"  controls preload></audio></td>
    <td><audio src="audio/grapheme-kaldi-emb-vesta.wav?raw=true"  controls preload></audio></td>
  </tr>
</tbody>
</table>
