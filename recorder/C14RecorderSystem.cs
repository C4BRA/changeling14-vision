// Dataset recorder for changeling14-vision.
//
// This file is copied into Content.Client/Replay/C14/ of a replay client that is
// built only for making datasets (see changeling14_vision/replays/record.py). It
// does nothing unless /c14-recorder/job.txt exists in the client's user data.
//
// For each moment of the replay (every interval_s seconds) it follows a few
// living players, saves what their screen shows (the game view only, no UI) as
// a PNG, and writes a JSON answer key with every entity drawn in that view:
// prototype id, name, sprite box in image pixels, and whether a wall blocks it.
// It also writes what is on the floor: every grid in view with the floor tile
// and visible gas of each tile, and every decal (warning stripes, markings,
// dirt) with its box. Ghosts are hidden first, since a living player can't see
// them.

using System.Globalization;
using System.IO;
using System.IO.Compression;
using System.Linq;
using System.Numerics;
using System.Text;
using Content.Client.Atmos.EntitySystems;
using Content.Client.Decals;
using Content.Client.Replay.Spectator;
using Content.Client.Viewport;
using Content.Shared.Atmos;
using Content.Shared.Atmos.Components;
using Content.Shared.Atmos.EntitySystems;
using Content.Shared.Examine;
using Content.Shared.Ghost.Components;
using Content.Shared.Humanoid;
using Content.Shared.Maps;
using Content.Shared.Mobs.Components;
using Content.Shared.Mobs.Systems;
using Content.Shared.NPC.Components;
using Content.Shared.Silicons.Borgs.Components;
using Content.Shared.SubFloor;
using Robust.Client;
using Robust.Client.GameObjects;
using Robust.Client.Graphics;
using Robust.Client.Replays.Loading;
using Robust.Client.Replays.Playback;
using Robust.Client.State;
using Robust.Shared.ContentPack;
using Robust.Shared.Containers;
using Robust.Shared.Graphics;
using Robust.Shared.Map;
using Robust.Shared.Map.Components;
using Robust.Shared.Replays;
using Robust.Shared.Utility;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;

namespace Content.Client.Replay.C14;

public sealed partial class C14RecorderSystem : EntitySystem
{
    [Dependency] private IResourceManager _res = default!;
    [Dependency] private IReplayLoadManager _load = default!;
    [Dependency] private IReplayPlaybackManager _playback = default!;
    [Dependency] private IStateManager _states = default!;
    [Dependency] private IEyeManager _eye = default!;
    [Dependency] private IGameController _game = default!;
    [Dependency] private ReplaySpectatorSystem _spectator = default!;
    [Dependency] private SpriteSystem _sprite = default!;
    [Dependency] private SharedTransformSystem _transform = default!;
    [Dependency] private SharedContainerSystem _containers = default!;
    [Dependency] private ExamineSystemShared _examine = default!;
    [Dependency] private MobStateSystem _mobs = default!;
    [Dependency] private SharedMapSystem _maps = default!;
    [Dependency] private DecalSystem _decals = default!;
    [Dependency] private GasTileOverlaySystem _gasOverlay = default!;
    [Dependency] private ITileDefinitionManager _tileDefs = default!;

    private static readonly ResPath Root = new("/c14-recorder");
    private static readonly ResPath JobFile = Root / "job.txt";
    private static readonly ResPath OutDir = Root / "out";

    // Systems can be torn down and made again when a replay loads, so the
    // recorder's progress lives in statics.
    private static Job? _job;
    private static bool _loadRequested;
    private static Step _step = Step.Load;
    private static int _moment = -1;
    private static int[] _momentIndices = [];
    private static List<EntityUid> _subjects = new();
    private static int _subject;
    private static int _wait;
    private static int _pending;
    private static int _saved;
    private static Random _random = new(0);

    private enum Step
    {
        Load,
        NextMoment,
        SettleMoment,
        NextSubject,
        SettleSubject,
        WaitForPicture,
        Done,
    }

    private sealed class Job
    {
        public string Replay = "/c14-recorder/replay.zip";
        public double IntervalSeconds = 30;
        public int PlayersPerMoment = 2;
        public int SettleFrames = 12;
        public int MaxPictures = 100000;
        public double StartSeconds;
        public double EndSeconds = double.MaxValue;
        public int Seed = 1;
        public float VisionRange = 20f;
    }

    public override void FrameUpdate(float frameTime)
    {
        base.FrameUpdate(frameTime);

        if (_step == Step.Done)
            return;

        if (_job == null)
        {
            if (!_res.UserData.Exists(JobFile))
                return;
            _job = ReadJob();
            _random = new Random(_job.Seed);
            _res.UserData.CreateDir(OutDir);
            Progress("job read");
        }

        try
        {
            Advance(_job);
        }
        catch (Exception e)
        {
            Log.Error($"c14 recorder failed: {e}");
            WriteText(Root / "error.txt", e.ToString());
            Finish();
        }
    }

    private void Advance(Job job)
    {
        if (_step == Step.Load)
        {
            if (!_loadRequested)
            {
                _loadRequested = true;
                Progress("loading replay");
                var path = new ResPath(job.Replay);
                var zip = new ZipArchive(_res.UserData.OpenRead(path));
                _load.LoadAndStartReplay(new ReplayFileReaderZip(zip, ReplayConstants.ReplayZipFolder));
                return;
            }

            if (_playback.Replay is not { } loaded || _states.CurrentState is not IMainViewportState)
                return;

            _momentIndices = PickMoments(loaded, job);
            Progress($"replay loaded, {_momentIndices.Length} moments");
            _step = Step.NextMoment;
            return;
        }

        if (_playback.Replay is not { } replay)
        {
            // Playback stopped underneath us (for example the replay failed part way).
            WriteText(Root / "error.txt", "replay playback stopped before recording finished");
            Finish();
            return;
        }

        switch (_step)
        {
            case Step.NextMoment:
                _moment++;
                if (_moment >= _momentIndices.Length || _saved >= job.MaxPictures)
                {
                    Finish();
                    return;
                }
                _playback.SetIndex(_momentIndices[_moment]);
                _wait = job.SettleFrames;
                _step = Step.SettleMoment;
                return;

            case Step.SettleMoment:
                if (--_wait > 0)
                    return;
                _subjects = PickSubjects(job.PlayersPerMoment);
                _subject = -1;
                Progress($"moment {_moment + 1}/{_momentIndices.Length} at {replay.CurrentReplayTime}, {_subjects.Count} players");
                _step = Step.NextSubject;
                return;

            case Step.NextSubject:
                _subject++;
                if (_subject >= _subjects.Count)
                {
                    _step = Step.NextMoment;
                    return;
                }
                if (!Exists(_subjects[_subject]))
                    return;
                _spectator.SpectateEntity(_subjects[_subject]);
                _wait = job.SettleFrames;
                _step = Step.SettleSubject;
                return;

            case Step.SettleSubject:
                if (--_wait > 0)
                    return;
                Capture(job, replay, _subjects[_subject]);
                _wait = 0;
                _step = Step.WaitForPicture;
                return;

            case Step.WaitForPicture:
                // The picture is taken when the game view next draws. If the window
                // is hidden that may never happen, so give up on it after a while.
                if (_pending > 0 && ++_wait < 600)
                    return;
                if (_pending > 0)
                {
                    Log.Warning("c14 recorder: picture never arrived, skipping");
                    _pending = 0;
                }
                _step = Step.NextSubject;
                return;
        }
    }

    private static int[] PickMoments(ReplayData replay, Job job)
    {
        var picked = new List<int>();
        var next = job.StartSeconds;
        for (var i = 0; i < replay.Count; i++)
        {
            var t = replay.ReplayTime[i].TotalSeconds;
            if (t > job.EndSeconds)
                break;
            if (t < next)
                continue;
            picked.Add(i);
            next = t + job.IntervalSeconds;
        }
        return picked.ToArray();
    }

    private List<EntityUid> PickSubjects(int count)
    {
        var living = new List<EntityUid>();
        var query = EntityQueryEnumerator<HumanoidProfileComponent, MobStateComponent, TransformComponent>();
        while (query.MoveNext(out var uid, out _, out var mob, out var xform))
        {
            if (xform.MapID == MapId.Nullspace || !_mobs.IsAlive(uid, mob))
                continue;
            if (_containers.IsEntityOrParentInContainer(uid))
                continue;
            living.Add(uid);
        }

        // Shuffle with the job's seed so a rerun picks the same players.
        living.Sort((a, b) => a.Id.CompareTo(b.Id));
        for (var i = living.Count - 1; i > 0; i--)
        {
            var j = _random.Next(i + 1);
            (living[i], living[j]) = (living[j], living[i]);
        }
        return living.Take(count).ToList();
    }

    private void Capture(Job job, ReplayData replay, EntityUid subject)
    {
        if (_states.CurrentState is not IMainViewportState state)
            return;

        var viewport = state.Viewport.Viewport;
        var eye = _eye.CurrentEye;

        // World position -> pixel in the viewport's own picture (the screenshot),
        // which is the world -> screen matrix with the screen placement taken off.
        Matrix3x2.Invert(viewport.GetLocalToScreenMatrix(), out var screenToLocal);
        var worldToPicture = viewport.GetWorldToScreenMatrix() * screenToLocal;

        HideGhosts();
        var entities = Describe(job, eye, subject, worldToPicture);
        Matrix3x2.Invert(worldToPicture, out var pictureToWorld);
        var pictureSize = viewport.Size;
        var floor = DescribeFloor(job, eye, subject, worldToPicture, pictureToWorld, (int) pictureSize.X, (int) pictureSize.Y);
        var name = $"{_moment:D5}-{subject.Id}";
        var tick = replay.CurTick.Value;
        var time = replay.CurrentReplayTime.TotalSeconds;
        var eyeRotation = eye.Rotation.Degrees;
        var zoom = eye.Zoom.X;

        _pending++;
        viewport.Screenshot(image =>
        {
            try
            {
                using (var png = _res.UserData.Open(OutDir / (name + ".png"), FileMode.Create))
                {
                    image.SaveAsPng(png);
                }

                var json = KeyJson(name, tick, time, subject, eyeRotation, zoom,
                    image.Width, image.Height, entities, floor);
                WriteText(OutDir / (name + ".json"), json);
                _saved++;
            }
            finally
            {
                if (_pending > 0)
                    _pending--;
            }
        });
    }

    private sealed class Seen
    {
        public EntityUid Uid;
        public string Prototype = "";
        public string Name = "";
        public Vector2 Center;
        public Vector2 Min;
        public Vector2 Max;
        public int DrawDepth;
        public double Rotation;
        public bool Blocked;
        public bool Alive;
        public bool Humanoid;
        public bool UnderFloor;
        public bool Mob;
        public bool Borg;
        public string Factions = "";
    }

    // Ghosts are drawn for a replay's spectator but not for a living player.
    private void HideGhosts()
    {
        var query = EntityQueryEnumerator<GhostComponent, SpriteComponent>();
        while (query.MoveNext(out var uid, out _, out var sprite))
        {
            if (sprite.Visible)
                _sprite.SetVisible((uid, sprite), false);
        }
    }

    private List<Seen> Describe(Job job, IEye eye, EntityUid subject, Matrix3x2 worldToPicture)
    {
        var seen = new List<Seen>();
        var eyeMap = eye.Position;
        var query = EntityQueryEnumerator<SpriteComponent, TransformComponent, MetaDataComponent>();
        while (query.MoveNext(out var uid, out var sprite, out var xform, out var meta))
        {
            if (!sprite.Visible || xform.MapID != eyeMap.MapId)
                continue;
            if (_containers.IsEntityOrParentInContainer(uid))
                continue;

            var (worldPos, worldRot) = _transform.GetWorldPositionRotation(uid);
            if ((worldPos - eyeMap.Position).Length() > job.VisionRange)
                continue;

            var bounds = _sprite.CalculateBounds((uid, sprite), worldPos, worldRot, eye.Rotation);
            var corners = new[] { bounds.BottomLeft, bounds.BottomRight, bounds.TopLeft, bounds.TopRight }
                .Select(c => Vector2.Transform(c, worldToPicture))
                .ToArray();
            var min = new Vector2(corners.Min(c => c.X), corners.Min(c => c.Y));
            var max = new Vector2(corners.Max(c => c.X), corners.Max(c => c.Y));
            if (max.X - min.X < 1 || max.Y - min.Y < 1)
                continue;

            var blocked = uid != subject && !CanSee(job, eyeMap, uid, subject, worldPos, bounds);

            seen.Add(new Seen
            {
                Uid = uid,
                Prototype = meta.EntityPrototype?.ID ?? "",
                Name = meta.EntityName,
                Center = Vector2.Transform(worldPos, worldToPicture),
                Min = min,
                Max = max,
                DrawDepth = sprite.DrawDepth,
                Rotation = worldRot.Degrees,
                Blocked = blocked,
                Alive = TryComp<MobStateComponent>(uid, out var mob) && _mobs.IsAlive(uid, mob),
                Humanoid = HasComp<HumanoidProfileComponent>(uid),
                UnderFloor = TryComp<SubFloorHideComponent>(uid, out var subfloor) && subfloor.IsUnderCover,
                Mob = HasComp<MobStateComponent>(uid),
                Borg = HasComp<BorgChassisComponent>(uid),
                Factions = TryComp<NpcFactionMemberComponent>(uid, out var factions)
                    ? string.Join(",", factions.Factions.Select(f => f.ToString()))
                    : "",
            });
        }
        return seen;
    }

    // A wall's own centre is often hidden behind the next wall along, even when
    // its face is in plain view, so a sprite counts as seen when a line from the
    // eye reaches its centre or any point near its corners.
    private bool CanSee(Job job, MapCoordinates eye, EntityUid uid, EntityUid subject, Vector2 worldPos,
        Box2Rotated bounds)
    {
        var centre = bounds.Center;
        var points = new[] { worldPos, bounds.BottomLeft, bounds.BottomRight, bounds.TopLeft, bounds.TopRight };
        for (var i = 0; i < points.Length; i++)
        {
            var point = i == 0 ? points[i] : Vector2.Lerp(centre, points[i], 0.8f);
            if (_examine.InRangeUnOccluded(eye, new MapCoordinates(point, eye.MapId), job.VisionRange,
                    e => e == uid || e == subject))
                return true;
        }
        return false;
    }

    // Every grid in view: a matrix from picture pixels to the grid's own
    // coordinates (tile x = floor(local x / tile_size)), the floor tile of each
    // tile there, and the visible gases and fire on it. Then every decal in view.
    private string DescribeFloor(Job job, IEye eye, EntityUid subject, Matrix3x2 worldToPicture,
        Matrix3x2 pictureToWorld, int width, int height)
    {
        var eyeMap = eye.Position;
        var tileNames = new List<string>();
        var tileIndex = new Dictionary<string, int>();
        var gasNames = new List<string>();
        foreach (var id in _gasOverlay.VisibleGasId)
            gasNames.Add(((Gas) id).ToString());
        gasNames.Add("Fire");
        var decalNames = new List<string>();
        var decalIndex = new Dictionary<string, int>();

        var corners = new[] { new Vector2(0, 0), new Vector2(width, 0), new Vector2(0, height), new Vector2(width, height) }
            .Select(c => Vector2.Transform(c, pictureToWorld)).ToArray();

        var grids = new StringBuilder();
        var decals = new StringBuilder();
        var query = EntityQueryEnumerator<MapGridComponent, TransformComponent>();
        while (query.MoveNext(out var gridUid, out var grid, out var xform))
        {
            if (xform.MapID != eyeMap.MapId)
                continue;
            var worldToLocal = _transform.GetInvWorldMatrix(gridUid);
            var localToWorld = _transform.GetWorldMatrix(gridUid);
            var local = corners.Select(c => Vector2.Transform(c, worldToLocal)).ToArray();
            var box = new Box2(local.Min(c => c.X) - 1, local.Min(c => c.Y) - 1,
                local.Max(c => c.X) + 1, local.Max(c => c.Y) + 1);

            TryComp<GasTileOverlayComponent>(gridUid, out var gas);
            var tiles = new StringBuilder();
            var gases = new StringBuilder();
            var tilesEnum = _maps.GetLocalTilesIntersecting(gridUid, grid, box, true);
            while (tilesEnum.MoveNext(out var tileRef))
            {
                var tileId = _tileDefs[tileRef.Tile.TypeId].ID;
                if (!tileIndex.TryGetValue(tileId, out var t))
                {
                    t = tileNames.Count;
                    tileIndex[tileId] = t;
                    tileNames.Add(tileId);
                }
                var at = tileRef.GridIndices;
                if (tiles.Length > 0)
                    tiles.Append(',');
                tiles.Append('[').Append(at.X).Append(',').Append(at.Y).Append(',').Append(t).Append(']');

                if (gas == null || !gas.Chunks.TryGetValue(SharedGasTileOverlaySystem.GetGasChunkIndices(at), out var chunk))
                    continue;
                // GasOverlayChunk.GetDataIndex, which content outside the gas systems may not call.
                var size = SharedGasTileOverlaySystem.ChunkSize;
                var data = chunk.TileData[(at.X - chunk.Origin.X) + (at.Y - chunk.Origin.Y) * size];
                if (data.Opacity != null)
                {
                    for (var g = 0; g < data.Opacity.Length && g < gasNames.Count - 1; g++)
                    {
                        if (data.Opacity[g] == 0)
                            continue;
                        if (gases.Length > 0)
                            gases.Append(',');
                        gases.Append('[').Append(at.X).Append(',').Append(at.Y).Append(',').Append(g).Append(',')
                            .Append(data.Opacity[g]).Append(']');
                    }
                }
                if (data.FireState != 0)
                {
                    if (gases.Length > 0)
                        gases.Append(',');
                    gases.Append('[').Append(at.X).Append(',').Append(at.Y).Append(',').Append(gasNames.Count - 1)
                        .Append(',').Append(data.FireState).Append(']');
                }
            }
            if (tiles.Length == 0)
                continue;

            var m = worldToLocal;
            var p = pictureToWorld * m;
            if (grids.Length > 0)
                grids.Append(',');
            grids.Append("{\"uid\":").Append(gridUid.Id);
            grids.Append(",\"m\":[").Append(Num6(p.M11)).Append(',').Append(Num6(p.M12)).Append(',')
                .Append(Num6(p.M21)).Append(',').Append(Num6(p.M22)).Append(',')
                .Append(Num6(p.M31)).Append(',').Append(Num6(p.M32)).Append(']');
            grids.Append(",\"tile_size\":").Append(grid.TileSize);
            grids.Append(",\"tiles\":[").Append(tiles).Append(']');
            grids.Append(",\"gas\":[").Append(gases).Append("]}");

            // Decals sit at their bottom-left corner and cover one tile.
            var localToPicture = localToWorld * worldToPicture;
            foreach (var (_, decal) in _decals.GetDecalsIntersecting(gridUid, box))
            {
                var c = decal.Coordinates;
                var pts = new[] { c, c + new Vector2(1, 0), c + new Vector2(0, 1), c + new Vector2(1, 1) }
                    .Select(v => Vector2.Transform(v, localToPicture)).ToArray();
                var min = new Vector2(pts.Min(v => v.X), pts.Min(v => v.Y));
                var max = new Vector2(pts.Max(v => v.X), pts.Max(v => v.Y));
                if (max.X < 0 || max.Y < 0 || min.X > width || min.Y > height)
                    continue;
                var centreWorld = Vector2.Transform(c + new Vector2(0.5f, 0.5f), localToWorld);
                var blocked = !_examine.InRangeUnOccluded(eyeMap, new MapCoordinates(centreWorld, eyeMap.MapId),
                    job.VisionRange, e => e == subject);
                if (!decalIndex.TryGetValue(decal.Id, out var d))
                {
                    d = decalNames.Count;
                    decalIndex[decal.Id] = d;
                    decalNames.Add(decal.Id);
                }
                if (decals.Length > 0)
                    decals.Append(',');
                decals.Append("{\"id\":").Append(d);
                decals.Append(",\"box\":[").Append(Num(min.X)).Append(',').Append(Num(min.Y)).Append(',')
                    .Append(Num(max.X)).Append(',').Append(Num(max.Y)).Append(']');
                decals.Append(",\"z\":").Append(decal.ZIndex);
                decals.Append(",\"blocked\":").Append(blocked ? "true" : "false").Append('}');
            }
        }

        var sb = new StringBuilder();
        sb.Append("\"tile_names\":[").Append(string.Join(",", tileNames.Select(Str))).Append(']');
        sb.Append(",\"gas_names\":[").Append(string.Join(",", gasNames.Select(Str))).Append(']');
        sb.Append(",\"decal_names\":[").Append(string.Join(",", decalNames.Select(Str))).Append(']');
        sb.Append(",\"grids\":[").Append(grids).Append(']');
        sb.Append(",\"decals\":[").Append(decals).Append(']');
        return sb.ToString();
    }

    private static string KeyJson(string name, uint tick, double time, EntityUid subject,
        double eyeRotation, float zoom, int width, int height, List<Seen> entities, string floor)
    {
        var sb = new StringBuilder();
        sb.Append('{');
        sb.Append("\"picture\":").Append(Str(name + ".png"));
        sb.Append(",\"tick\":").Append(tick);
        sb.Append(",\"time\":").Append(Num(time));
        sb.Append(",\"subject\":").Append(subject.Id);
        sb.Append(",\"eye_rotation\":").Append(Num(eyeRotation));
        sb.Append(",\"zoom\":").Append(Num(zoom));
        sb.Append(",\"width\":").Append(width);
        sb.Append(",\"height\":").Append(height);
        sb.Append(',').Append(floor);
        sb.Append(",\"entities\":[");
        var first = true;
        foreach (var e in entities)
        {
            // Only keep sprites that land at least partly inside the picture.
            if (e.Max.X < 0 || e.Max.Y < 0 || e.Min.X > width || e.Min.Y > height)
                continue;
            if (!first)
                sb.Append(',');
            first = false;
            sb.Append("{\"uid\":").Append(e.Uid.Id);
            sb.Append(",\"prototype\":").Append(Str(e.Prototype));
            sb.Append(",\"name\":").Append(Str(e.Name));
            sb.Append(",\"center\":[").Append(Num(e.Center.X)).Append(',').Append(Num(e.Center.Y)).Append(']');
            sb.Append(",\"box\":[").Append(Num(e.Min.X)).Append(',').Append(Num(e.Min.Y)).Append(',')
                .Append(Num(e.Max.X)).Append(',').Append(Num(e.Max.Y)).Append(']');
            sb.Append(",\"draw_depth\":").Append(e.DrawDepth);
            sb.Append(",\"rotation\":").Append(Num(e.Rotation));
            sb.Append(",\"blocked\":").Append(e.Blocked ? "true" : "false");
            sb.Append(",\"alive\":").Append(e.Alive ? "true" : "false");
            sb.Append(",\"humanoid\":").Append(e.Humanoid ? "true" : "false");
            sb.Append(",\"under_floor\":").Append(e.UnderFloor ? "true" : "false");
            sb.Append(",\"mob\":").Append(e.Mob ? "true" : "false");
            sb.Append(",\"borg\":").Append(e.Borg ? "true" : "false");
            sb.Append(",\"factions\":").Append(Str(e.Factions));
            sb.Append(",\"subject\":").Append(e.Uid == subject ? "true" : "false");
            sb.Append('}');
        }
        sb.Append("]}");
        return sb.ToString();
    }

    private static string Num(double value)
    {
        return Math.Round(value, 2).ToString(CultureInfo.InvariantCulture);
    }

    private static string Num6(float value)
    {
        return Math.Round(value, 6).ToString(CultureInfo.InvariantCulture);
    }

    private static string Str(string value)
    {
        var sb = new StringBuilder("\"");
        foreach (var c in value)
        {
            switch (c)
            {
                case '"': sb.Append("\\\""); break;
                case '\\': sb.Append("\\\\"); break;
                case '\n': sb.Append("\\n"); break;
                case '\r': sb.Append("\\r"); break;
                case '\t': sb.Append("\\t"); break;
                default:
                    if (c < ' ')
                        sb.Append("\\u").Append(((int) c).ToString("x4", CultureInfo.InvariantCulture));
                    else
                        sb.Append(c);
                    break;
            }
        }
        return sb.Append('"').ToString();
    }

    private Job ReadJob()
    {
        var job = new Job();
        using var reader = new StreamReader(_res.UserData.OpenRead(JobFile));
        while (reader.ReadLine() is { } line)
        {
            var eq = line.IndexOf('=');
            if (line.StartsWith('#') || eq < 0)
                continue;
            var key = line[..eq].Trim();
            var value = line[(eq + 1)..].Trim();
            var inv = CultureInfo.InvariantCulture;
            switch (key)
            {
                case "replay": job.Replay = value; break;
                case "interval_s": job.IntervalSeconds = double.Parse(value, inv); break;
                case "players_per_moment": job.PlayersPerMoment = int.Parse(value, inv); break;
                case "settle_frames": job.SettleFrames = int.Parse(value, inv); break;
                case "max_pictures": job.MaxPictures = int.Parse(value, inv); break;
                case "start_s": job.StartSeconds = double.Parse(value, inv); break;
                case "end_s": job.EndSeconds = double.Parse(value, inv); break;
                case "seed": job.Seed = int.Parse(value, inv); break;
                case "vision_range": job.VisionRange = float.Parse(value, inv); break;
            }
        }
        return job;
    }

    private void Progress(string text)
    {
        Log.Info($"c14 recorder: {text}");
        WriteText(Root / "progress.txt", $"{text}\nsaved={_saved}\n");
    }

    private void WriteText(ResPath path, string text)
    {
        using var stream = _res.UserData.Open(path, FileMode.Create);
        using var writer = new StreamWriter(stream);
        writer.Write(text);
    }

    private void Finish()
    {
        _step = Step.Done;
        WriteText(Root / "done.txt", $"saved={_saved}\n");
        _game.Shutdown("c14 recorder finished");
    }
}
